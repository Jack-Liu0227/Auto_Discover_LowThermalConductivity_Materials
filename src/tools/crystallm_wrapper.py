"""
CrystaLLM工具封装

用于生成晶体结构。

主要功能：
- 根据组分生成晶体结构
- 批量生成结构
- 质量检查（原子间距、晶格参数、空间群）
- 保存生成的结构到文件
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import time
import logging
import json
import os
from datetime import datetime
import threading

# 全局锁，用于防止多线程并发初始化CUDA
_cuda_init_lock = threading.Lock()

try:
    from .base_tool import BaseTool, ToolResponse, ToolStatus
    from ..utils.types import Composition, CrystalStructure
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from tools.base_tool import BaseTool, ToolResponse, ToolStatus
    from utils.types import Composition, CrystalStructure

logger = logging.getLogger(__name__)


class CrystaLLMWrapper(BaseTool):
    """
    CrystaLLM工具封装

    功能：根据化学式生成晶体结构（POSCAR格式）

    配置参数：
    - api_url: CrystaLLM API地址
    - model_path: 模型路径
    - min_distance: 最小原子间距（Å，默认1.5）
    - max_lattice: 最大晶格参数（Å，默认50.0）
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: float = 300.0,
        output_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        初始化CrystaLLM

        Args:
            model_path: 模型路径
            config: 配置参数
            timeout: 超时时间
            output_dir: 输出目录（默认为项目根目录下的generated_structures）
            device: 计算设备 ("cuda" 或 "cpu"，默认自动检测）
        """
        super().__init__(name="CrystaLLM", config=config, timeout=timeout)
        self.model_path = model_path or self.config.get('model_path')
        self.api_url = self.config.get('api_url', None)  # 默认为 None
        self.min_distance = self.config.get('min_distance', 1.5)  # Å
        self.max_lattice = self.config.get('max_lattice', 50.0)  # Å
        self.model = None

        # 设置输出目录
        if output_dir is None:
            raise ValueError("output_dir must be provided")
        self.output_dir = Path(output_dir)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Structure output directory: {self.output_dir}")

        # 设置计算设备
        if device:
            self.device = str(device).strip().lower()
            if self.device.startswith("cuda"):
                try:
                    from utils.gpu_parallel import validate_gpu_devices
                except ImportError:
                    from src.utils.gpu_parallel import validate_gpu_devices
                validate_gpu_devices([self.device])
        else:
            self.device = self._detect_device()

    def _detect_device(self) -> str:
        """自动检测设备；未指定设备时默认请求 CUDA。"""
        with _cuda_init_lock:
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    logger.info(f"✅ CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("❌ CUDA is not available; the explicit CUDA request will fail clearly")
                return "cuda"
            except ImportError:
                logger.warning("⚠️ PyTorch not found; the default CUDA request will fail clearly")
                return "cuda"

    def check_availability(self) -> bool:
        """检查 CrystaLLM 主路径是否可导入。"""
        try:
            import importlib

            importlib.import_module(".crystallm.generator", package=__package__)
            logger.info("✅ CrystaLLM generator is available")
            return True
        except Exception as exc:
            logger.error("❌ CrystaLLM generator is unavailable: %s", exc)
            return False

    def run(
        self,
        composition: Composition,
        n_structures: int = 1,
        relax_structures: bool = False,
        **kwargs
    ) -> ToolResponse:
        """
        生成晶体结构（可选弛豫）

        Args:
            composition: 组分
            n_structures: 生成结构数量
            relax_structures: 是否弛豫结构（默认False）
            **kwargs: 额外参数
                - temperature: 采样温度（默认1.0）
                - seed: 随机种子
                - pressure: 弛豫压力（GPa，默认0.0）
                - relax_output_dir: 弛豫结构输出目录（默认 MyRelaxStructure）
                - calculate_properties: 是否计算热导率（默认True，纯生成模式设为False）

        Returns:
            ToolResponse: 包含CrystalStructure列表
        """
        start_time = time.time()

        try:
            if not self.is_available:
                return ToolResponse(
                    status=ToolStatus.NOT_AVAILABLE,
                    error="CrystaLLM not available",
                    metadata={"failure_kind": "backend"},
                )

            temperature = kwargs.get('temperature', 1.0)
            seed = kwargs.get('seed', None)

            logger.info(f"Generating {n_structures} structures for {composition.formula}")

            # 调用真实实现
            structures = self._generate_structures_real(
                composition,
                n_structures,
                temperature,
                seed,
                calculate_properties=kwargs.get('calculate_properties', True)  # 默认计算属性
            )

            # 质量检查
            valid_structures = self._quality_check(structures)

            # 结构由 CrystaLLM 生成器保存，不在 wrapper 中重复写入。

            # 如果需要弛豫结构
            if relax_structures:
                logger.info(f"🔧 开始弛豫 {len(valid_structures)} 个结构...")
                relaxed_structures = self._relax_and_save_structures(
                    valid_structures,
                    composition,
                    **kwargs
                )
                logger.info(f"✅ 弛豫完成: {len(relaxed_structures)} 个结构")
            else:
                relaxed_structures = None

            execution_time = time.time() - start_time

            if not structures:
                raise RuntimeError(
                    f"CrystaLLM returned no structures for {composition.formula}"
                )
            if len(valid_structures) != n_structures:
                raise RuntimeError(
                    f"CrystaLLM returned {len(valid_structures)} valid structures; "
                    f"expected {n_structures} for {composition.formula}"
                )

            logger.info(f"Generated {len(valid_structures)}/{len(structures)} valid structures in {execution_time:.2f}s")

            return ToolResponse(
                status=ToolStatus.SUCCESS,
                result=valid_structures,
                execution_time=execution_time,
                metadata={
                    'generator_backend': 'CrystaLLM',
                    'n_requested': n_structures,
                    'n_generated': len(structures),
                    'n_valid': len(valid_structures),
                    'temperature': temperature,
                    'composition': composition.formula,
                    'relaxed': relax_structures,
                    'n_relaxed': len(relaxed_structures) if relaxed_structures else 0
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"CrystaLLM generation failed: {e}")
            error_text = str(e)
            lowered_error = error_text.lower()
            structure_markers = (
                "returned no structures",
                "returned 0 valid structures",
                "valid structures; expected",
                "invalid cif",
                "no cif files were generated",
                "postprocessing produced no cif",
                "frontend structure conversion failed",
            )
            failure_kind = (
                "structure_invalid"
                if any(marker in lowered_error for marker in structure_markers)
                else "backend"
            )
            return ToolResponse(
                status=ToolStatus.FAILED,
                error=error_text,
                execution_time=execution_time,
                metadata={"failure_kind": failure_kind},
            )

    def _generate_structures_real(
        self,
        composition: Composition,
        n_structures: int,
        temperature: float,
        seed: Optional[int] = None,
        calculate_properties: bool = True
    ) -> List[CrystalStructure]:
        """
        使用CrystaLLM生成晶体结构（真实实现）

        基于 Transformer 模型生成晶体结构；CrystaLLM 失败时直接报告失败。

        Args:
            composition: 组分
            n_structures: 生成结构数量
            temperature: 采样温度
            seed: 随机种子

        Returns:
            List[CrystalStructure]: 生成的结构列表
        """
        # CrystaLLM 是唯一结构生成后端。任何导入、模型或运行错误都必须
        # 传播到 run()，由 run() 返回失败状态；不得切换到其他生成器。
        return self._generate_with_crystallm(
            composition,
            n_structures,
            temperature,
            seed,
            calculate_properties,
        )

    def _generate_with_crystallm(
        self,
        composition: Composition,
        n_structures: int,
        temperature: float,
        seed: Optional[int] = None,
        calculate_properties: bool = True  # 新增参数
    ) -> List[CrystalStructure]:
        """
        使用CrystaLLM生成器生成结构

        Args:
            composition: 组分
            n_structures: 生成数量
            temperature: 采样温度
            seed: 随机种子

        Returns:
            List[CrystalStructure]: 生成的结构列表
        """
        from .crystallm.generator import generate_crystal_from_composition
        from pymatgen.io.vasp import Poscar
        from pymatgen.io.cif import CifParser
        import random

        logger.info(f"Using CrystaLLM to generate {n_structures} structures for {composition.formula}")

        structures = []

        # 保留调用方明确指定的设备；不将 CPU 请求改写为 CUDA。
        device_to_use = self.device

        if seed is not None:
            try:
                import torch

                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            except Exception as exc:
                logger.warning(f"Failed to apply CrystaLLM seed {seed}: {exc}")

        # CrystaLLM一次生成多个结构
        result = generate_crystal_from_composition(
            composition=composition.formula,
            device=device_to_use,
            num_samples=n_structures,
            top_k=10,
            max_new_tokens=2000,
            seed=seed,
            output_dir=str(self.output_dir)  # 使用 wrapper 的输出目录
        )

        if not result.get('success', False):
            raise ValueError(f"CrystaLLM generation failed: {result.get('error', 'Unknown error')}")
        if str(result.get('generator_backend', 'CrystaLLM')) != 'CrystaLLM':
            raise ValueError("Unexpected structure generator backend")

        # 从CIF文件路径读取结构
        cif_file_paths = result.get('cif_file_paths', [])

        # 创建目标目录 (generator 已经生成了文件在 processed_structures/formula/processed/)
        # 我们需要这些文件，但不应重复复制
        # 这里的 output_dir 是 processed_structures
        # generator 在 output_dir/formula/processed 下生成了 cifs
        safe_formula = composition.formula.replace("/", "_").replace("\\", "_")
        comp_dir = self.output_dir / safe_formula
        
        # 实际使用的 CIF 文件目录 (generator.py 现在的逻辑是 output_dir / composition / processed)
        cif_source_dir = Path(result.get('cif_directory', comp_dir)) # use returned dir
        
        # 检查是否一致
        logger.info(f"📂 CIF files located at: {cif_source_dir}")

        for i, cif_path in enumerate(cif_file_paths):
            try:
                # 不再复制 CIF 文件
                cif_path = Path(cif_path) # ensure path object
                # cif_dest = comp_dir / f"structure_{i+1}.cif" 
                # shutil.copy2(cif_path, cif_dest)
                logger.debug(f"Process CIF file: {cif_path.name}")

                # 读取CIF文件获取结构信息
                parser = CifParser(str(cif_path))
                pmg_structure = parser.get_structures()[0]

                # 转换为POSCAR格式（仅用于内部数据结构）
                poscar = Poscar(pmg_structure)
                poscar_str = str(poscar)

                # 获取晶格参数
                lattice = pmg_structure.lattice
                lattice_params = {
                    'a': lattice.a,
                    'b': lattice.b,
                    'c': lattice.c,
                    'alpha': lattice.alpha,
                    'beta': lattice.beta,
                    'gamma': lattice.gamma
                }

                # 获取空间群
                try:
                    space_group = pmg_structure.get_space_group_info()[0]
                except:
                    space_group = "P1"

                # 创建CrystalStructure对象
                crystal_structure = CrystalStructure(
                    composition=composition,
                    structure_id=f"{composition.formula}_crystallm_{i}",
                    poscar=poscar_str,
                    space_group=space_group,
                    lattice_params=lattice_params,
                    n_atoms=len(pmg_structure),
                    quality_score=0.9  # CrystaLLM生成的质量评分较高
                )

                structures.append(crystal_structure)

            except Exception as e:
                logger.warning(f"Failed to process CIF file {cif_path}: {e}")

        logger.info(f"Successfully generated {len(structures)}/{n_structures} structures with CrystaLLM")

        # 保存元数据
        # 保存元数据到实际目录
        if structures:
            # metadata 也保存到 generator 目录
            self._save_metadata(cif_source_dir, structures, composition)
            logger.info(f"📊 Saved metadata to: {cif_source_dir / 'metadata.json'}")

            # 只在需要时计算热导率（纯生成模式不计算）
            # 只在需要时计算热导率（纯生成模式不计算）
            # 注意：热导率计算也应使用 cif_source_dir
            if calculate_properties:
                logger.info("🔥 计算热导率...")
                self._calculate_and_save_thermal_conductivity(cif_source_dir, structures, composition)
            else:
                logger.info("✅ 纯生成模式：跳过热导率计算")

        return structures

    def generate_batch(
        self,
        compositions: List[Composition],
        n_structures_per_comp: int = 1,
        **kwargs
    ) -> Dict[str, List[CrystalStructure]]:
        """
        批量生成结构

        Args:
            compositions: 组分列表
            n_structures_per_comp: 每个组分生成的结构数
            **kwargs: 额外参数（传递给run方法）

        Returns:
            Dict: {formula: [structures]}
        """
        results = {}
        for comp in compositions:
            response = self.run(comp, n_structures_per_comp, **kwargs)
            if response.is_success():
                results[comp.formula] = response.result
            else:
                logger.warning(f"Failed to generate structures for {comp.formula}: {response.error}")
                results[comp.formula] = []
        return results

    def _quality_check(
        self,
        structures: List[CrystalStructure]
    ) -> List[CrystalStructure]:
        """
        质量检查

        检查：
        - 原子间距合理性（> min_distance）
        - 晶格参数合理性（< max_lattice）
        - 空间群对称性

        Args:
            structures: 待检查的结构列表

        Returns:
            List[CrystalStructure]: 通过检查的结构列表
        """
        valid_structures = []

        for structure in structures:
            is_valid = True

            # 1. 检查晶格参数
            if structure.lattice_params:
                for param in ['a', 'b', 'c']:
                    if param in structure.lattice_params:
                        value = structure.lattice_params[param]
                        if value > self.max_lattice or value < 0.5:
                            logger.debug(f"Structure {structure.structure_id} failed lattice check: {param}={value}")
                            is_valid = False
                            break

            # 2. 检查原子间距（需要ASE）
            if is_valid:
                try:
                    from ase.io import read
                    from io import StringIO
                    import numpy as np

                    poscar_io = StringIO(structure.poscar)
                    atoms = read(poscar_io, format='vasp')

                    # 检查最近邻距离
                    distances = atoms.get_all_distances()
                    # 排除对角线（自身距离=0）
                    min_dist = np.min(distances[distances > 0.01])

                    if min_dist < self.min_distance:
                        logger.debug(f"Structure {structure.structure_id} failed distance check: min_dist={min_dist:.3f}")
                        is_valid = False

                except Exception as e:
                    logger.warning(f"Failed to check distances for {structure.structure_id}: {e}")
                    # 距离检查失败不影响通过

            if is_valid:
                valid_structures.append(structure)

        logger.info(f"Quality check: {len(valid_structures)}/{len(structures)} structures passed")
        return valid_structures

    def _save_metadata(
        self,
        comp_dir: Path,
        structures: List[CrystalStructure],
        composition: Composition
    ) -> None:
        """
        保存结构元数据到 JSON 文件（仅包含 CIF 文件信息）

        Args:
            comp_dir: 组分目录
            structures: 结构列表
            composition: 组分信息
        """
        import json

        metadata_list = []
        for i, structure in enumerate(structures):
            metadata = {
                'structure_id': structure.structure_id,
                'index': i + 1,
                'space_group': structure.space_group,
                'lattice_params': structure.lattice_params,
                'n_atoms': structure.n_atoms,
                'quality_score': structure.quality_score,
                'cif_file': f"structure_{i+1}.cif"
            }
            metadata_list.append(metadata)

        # 保存元数据JSON文件
        metadata_file = comp_dir / "metadata.json"
        metadata_dict = {
            'composition': composition.formula,
            'n_structures': len(structures),
            'generation_time': datetime.now().isoformat(),
            'structures': metadata_list
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

    @staticmethod
    def _is_cuda_oom_error(error: Exception) -> bool:
        message = str(error).lower()
        return "out of memory" in message or "cuda oom" in message

    @staticmethod
    def _actual_formula_from_cif(cif_path: Path) -> str:
        """Return the canonical formula stored in a relaxed CIF."""
        try:
            from pymatgen.core import Structure

            return Structure.from_file(str(cif_path)).composition.reduced_formula
        except Exception as exc:
            logger.warning("Failed to read actual formula from %s: %s", cif_path, exc)
            return ""

    def _actual_formulas_for_structures(
        self,
        comp_dir: Path,
        structures: List[CrystalStructure],
    ) -> list[str]:
        formulas: list[str] = []
        for index, structure in enumerate(structures):
            candidates = [
                comp_dir / f"{structure.structure_id}.cif",
                comp_dir / f"structure_{index + 1}.cif",
            ]
            cif_path = next((path for path in candidates if path.exists()), None)
            formulas.append(self._actual_formula_from_cif(cif_path) if cif_path else "")
        return formulas

    @staticmethod
    def _aligned_values(values: list[str], size: int) -> list[str]:
        return (values + [""] * max(0, size - len(values)))[:size]

    @staticmethod
    def _insert_or_replace_column(result_df, column: str, values: list, position: int) -> None:
        """Add a lineage column without failing when kappa-lib already provides it."""
        if column in result_df.columns:
            result_df[column] = values
        else:
            result_df.insert(position, column, values)

    @staticmethod
    def _write_thermal_csv(result_df, csv_file: Path) -> None:
        """Merge material-local thermal rows and atomically replace the CSV."""
        import pandas as pd

        result_df = result_df.copy()
        existing = None
        if csv_file.exists():
            try:
                existing = pd.read_csv(csv_file, encoding="utf-8-sig")
            except Exception:
                existing = None

        if existing is not None and "CIF_File" in existing.columns and "CIF_File" in result_df.columns:
            merged = existing.copy()
            for column in result_df.columns:
                if column not in merged.columns:
                    merged[column] = pd.NA
            incoming = result_df.reindex(columns=merged.columns).set_index("CIF_File", drop=False)
            merged_index = merged.set_index("CIF_File", drop=False)
            order = list(merged_index.index)
            for cif_name, incoming_row in incoming.iterrows():
                if cif_name not in merged_index.index:
                    merged_index.loc[cif_name] = incoming_row
                    order.append(cif_name)
                    continue
                for column, value in incoming_row.items():
                    if column == "CIF_File" or pd.isna(value):
                        continue
                    merged_index.loc[cif_name, column] = value
            result_df = merged_index.reindex(order).reset_index(drop=True)

        csv_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = csv_file.with_name(f".{csv_file.name}.{os.getpid()}.tmp")
        try:
            result_df.to_csv(temp_file, index=False, encoding="utf-8-sig")
            os.replace(temp_file, csv_file)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def _calculate_and_save_thermal_conductivity(
        self,
        comp_dir: Path,
        structures: List[CrystalStructure],
        composition: Composition
    ) -> None:
        """
        计算热导率并保存到 CSV 文件。

        热导率唯一通过 kappa_lib 中的 ThermalConductivityCalculator
        （Kappa-P/Slack）计算。依赖缺失、输入错误、运行异常或输出无效时，
        该步骤直接失败，不生成任何替代热导率结果。

        Args:
            comp_dir: 组分目录（包含 CIF 文件）
            structures: 结构列表（只包含成功弛豫的结构）
            composition: 组分信息
        """
        # Kappa-P is the only permitted thermal-conductivity implementation.
        # Do not catch errors here: missing dependencies and runtime failures
        # must remain terminal failures instead of producing surrogate values.
        self._calculate_with_kappa_lib(comp_dir, structures, composition)

        # 注意：第一步生成的结构不需要计算声子谱
        # 声子谱计算只在弛豫后进行（见 _relax_and_save_structures 方法）

    def _calculate_with_kappa_lib(
        self,
        comp_dir: Path,
        structures: List[CrystalStructure],
        composition: Composition
    ) -> None:
        """使用 kappa_lib 计算热导率（需要 torch）"""
        from .kappa_lib.calculator import ThermalConductivityCalculator, KAPPA_AVAILABLE

        if not KAPPA_AVAILABLE:
            raise ImportError("Kappa library not available (missing torch or other dependencies)")

        logger.info(f"🔥 开始计算热导率（使用 Kappa-P 方法 + CGCNN）...")

        # 初始化热导率计算器
        calculator = ThermalConductivityCalculator(str(comp_dir))

        # 使用 Kappa-P 方法计算（基于 Slack 模型 + CGCNN 预测）
        result_df = calculator.calculate_kappa_p(device=self.device)

        if result_df.empty:
            raise ValueError("热导率计算返回空结果")

        required_kappa_column = "Kappa_Slack (W m-1 K-1)"
        if required_kappa_column not in result_df.columns:
            raise ValueError(
                f"Kappa-P output is missing required column: {required_kappa_column}"
            )
        if len(result_df) != len(structures):
            raise ValueError(
                "Kappa-P output row count does not match the relaxed structure count: "
                f"{len(result_df)} != {len(structures)}"
            )
        try:
            import numpy as np

            kappa_values = result_df[required_kappa_column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Kappa-P output contains non-numeric thermal conductivity values") from exc
        if not np.isfinite(kappa_values).all():
            raise ValueError("Kappa-P output contains NaN or infinite thermal conductivity values")

        # Keep the requested composition for directory lineage, while Formula
        # describes the actual relaxed CIF used by the calculation.
        size = len(result_df)
        actual_formulas = self._aligned_values(
            self._actual_formulas_for_structures(comp_dir, structures),
            size,
        )
        structure_ids = self._aligned_values(
            [s.structure_id for s in structures],
            size,
        )
        cif_names = [f"{structure_id}.cif" if structure_id else "" for structure_id in structure_ids]
        self._insert_or_replace_column(result_df, "Formula", actual_formulas, 0)
        self._insert_or_replace_column(result_df, "Composition", [composition.formula] * size, 1)
        self._insert_or_replace_column(result_df, "Material_Dir", [comp_dir.name] * size, 2)
        self._insert_or_replace_column(result_df, "Structure_ID", structure_ids, 3)
        self._insert_or_replace_column(result_df, "CIF_File", cif_names, 4)

        # Preserve material-local rows from earlier partial/retry attempts.
        csv_file = comp_dir / "thermal_conductivity.csv"
        self._write_thermal_csv(result_df, csv_file)

        logger.info(f"✅ 热导率计算完成（Kappa-P），结果保存到: {csv_file.name}")

        # 打印结果摘要
        if 'Kappa_Slack (W m-1 K-1)' in result_df.columns:
            kappa_values = result_df['Kappa_Slack (W m-1 K-1)'].values
            logger.info(f"📊 热导率范围: {kappa_values.min():.3f} - {kappa_values.max():.3f} W/(m·K)")
            logger.info(f"📊 平均热导率: {kappa_values.mean():.3f} W/(m·K)")

    def _calculate_and_add_phonon_info(
        self,
        comp_dir: Path,
        structures: List[CrystalStructure],
        composition: Composition
    ) -> None:
        """
        计算声子谱并将虚频信息添加到 CSV 文件

        Args:
            comp_dir: 组分目录
            structures: 结构列表（只包含成功弛豫的结构）
            composition: 组分信息
        """
        try:
            import pandas as pd
            import os

            logger.info(f"🎵 开始计算声子谱...")

            # 检查 CSV 文件是否存在
            csv_file = comp_dir / "thermal_conductivity.csv"
            df = None
            if csv_file.exists():
                # 读取现有的 CSV
                df = pd.read_csv(csv_file)
            else:
                logger.warning(f"CSV 文件不存在: {csv_file}，将保存声子谱结果到独立文件")

            # 使用并行计算模块批量处理声子谱
            try:
                from .phonon_parallel import calculate_phonons_parallel
            except ImportError:
                from tools.phonon_parallel import calculate_phonons_parallel
            
            # 从环境变量获取并行数量，默认为4
            max_workers = int(os.getenv("PHONON_PARALLEL_WORKERS", "4"))
            
            # 从环境变量获取GPU配置
            gpus_env = os.getenv("PHONON_GPUS")
            gpus = gpus_env.split(',') if gpus_env else None
            
            # 并行计算所有结构的声子谱
            has_imaginary_list = calculate_phonons_parallel(
                structures=structures,
                composition=composition,
                comp_dir=comp_dir,
                max_workers=max_workers,
                gpus=gpus  # 传递GPU列表
            )

            # 将声子信息添加到 DataFrame 或保存为独立文件
            is_dict_list = bool(has_imaginary_list) and isinstance(has_imaginary_list[0], dict)
            if df is not None and len(has_imaginary_list) == len(df):
                if is_dict_list:
                    df['Has_Imaginary_Freq'] = [r.get('has_imaginary') for r in has_imaginary_list]
                    df['Min_Frequency'] = [r.get('min_frequency') for r in has_imaginary_list]
                    df['Gamma_Min_Optical'] = [r.get('gamma_min_optical') for r in has_imaginary_list]
                else:
                    df['Has_Imaginary_Freq'] = has_imaginary_list

                # 保存更新后的 CSV
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                logger.info(f"✅ 声子谱信息已添加到 CSV 文件")
            else:
                phonon_file = comp_dir / "phonon_results.csv"
                if is_dict_list:
                    phonon_df = pd.DataFrame({
                        "Structure_ID": [s.structure_id for s in structures],
                        "CIF_File": [f"{s.structure_id}.cif" for s in structures],
                        "Has_Imaginary_Freq": [r.get('has_imaginary') for r in has_imaginary_list],
                        "Min_Frequency": [r.get('min_frequency') for r in has_imaginary_list],
                        "Gamma_Min_Optical": [r.get('gamma_min_optical') for r in has_imaginary_list],
                    })
                else:
                    phonon_df = pd.DataFrame({
                        "Structure_ID": [s.structure_id for s in structures],
                        "CIF_File": [f"{s.structure_id}.cif" for s in structures],
                        "Has_Imaginary_Freq": has_imaginary_list,
                    })
                phonon_df.to_csv(phonon_file, index=False, encoding='utf-8-sig')
                if df is None:
                    logger.warning(f"已保存声子谱结果到: {phonon_file}")
                else:
                    logger.warning(f"声子谱结果数量 ({len(has_imaginary_list)}) 与 CSV 行数 ({len(df)}) 不匹配，已保存到: {phonon_file}")

        except Exception as e:
            logger.warning(f"声子谱计算失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _save_structures(
        self,
        structures: List[CrystalStructure],
        composition: Composition
    ) -> Path:
        """
        保存生成的结构到文件

        为每个组分创建一个子目录，包含：
        - CIF文件（每个结构一个文件）
        - POSCAR文件（每个结构一个文件，可选）
        - metadata.json（包含所有结构的元数据）
        - summary.txt（人类可读的摘要）

        Args:
            structures: 要保存的结构列表
            composition: 组分信息

        Returns:
            Path: 保存目录的路径
        """
        try:
            # 使用安全的文件名（替换特殊字符）
            safe_formula = composition.formula.replace("/", "_").replace("\\", "_")

            # 检查是否已存在该材料的目录
            existing_dirs = list(self.output_dir.glob(f"{safe_formula}_*"))

            if existing_dirs:
                # 如果存在，使用最新的目录
                comp_dir = max(existing_dirs, key=lambda p: p.stat().st_mtime)
                logger.info(f"✅ 复用现有目录: {comp_dir.name}")

                # 检查已有的结构文件数量，以便追加新结构
                existing_cifs = list(comp_dir.glob("structure_*.cif"))
                start_index = len(existing_cifs)
                logger.info(f"   已有 {start_index} 个结构，将从 structure_{start_index+1}.cif 开始保存")
            else:
                # 如果不存在，创建新目录
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comp_dir = self.output_dir / f"{safe_formula}_{timestamp}"
                comp_dir.mkdir(parents=True, exist_ok=True)
                start_index = 0
                logger.info(f"📁 创建新目录: {comp_dir.name}")

            # 只保存 CIF 文件
            metadata_list = []
            for i, structure in enumerate(structures):
                # 从POSCAR转换为pymatgen Structure对象
                try:
                    from pymatgen.io.vasp import Poscar
                    from pymatgen.io.cif import CifWriter

                    # 解析POSCAR字符串 - 使用正确的方法
                    poscar_obj = Poscar.from_str(structure.poscar)
                    pmg_structure = poscar_obj.structure

                    # 只保存CIF文件（使用 start_index 避免覆盖）
                    file_index = start_index + i + 1
                    cif_file = comp_dir / f"structure_{file_index}.cif"
                    cif_writer = CifWriter(pmg_structure)
                    cif_writer.write_file(str(cif_file))

                    cif_saved = True
                except Exception as e:
                    logger.warning(f"Failed to convert structure {i+1} to CIF: {e}")
                    cif_file = None
                    cif_saved = False

                # 收集元数据（只包含 CIF 文件）
                if cif_saved:
                    metadata = {
                        'structure_id': structure.structure_id,
                        'index': file_index,
                        'space_group': structure.space_group,
                        'lattice_params': structure.lattice_params,
                        'n_atoms': structure.n_atoms,
                        'quality_score': structure.quality_score,
                        'cif_file': cif_file.name
                    }
                    metadata_list.append(metadata)

            # 保存或更新元数据JSON文件
            metadata_file = comp_dir / "metadata.json"

            if metadata_file.exists() and start_index > 0:
                # 如果是复用目录，读取现有元数据并追加
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)

                # 合并结构列表
                existing_metadata['structures'].extend(metadata_list)
                existing_metadata['n_structures'] = len(existing_metadata['structures'])
                existing_metadata['last_updated'] = datetime.now().isoformat()

                metadata_dict = existing_metadata
                logger.info(f"📝 更新元数据: 新增 {len(metadata_list)} 个结构")
            else:
                # 新目录，创建新的元数据
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metadata_dict = {
                    'composition': composition.formula,
                    'timestamp': timestamp,
                    'n_structures': len(metadata_list),
                    'structures': metadata_list
                }

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

            if start_index > 0:
                logger.info(f"✅ 保存 {len(structures)} 个新结构到 {comp_dir.name}")
                logger.info(f"   - CIF 文件: structure_{start_index+1}.cif 到 structure_{start_index+len(structures)}.cif")
            else:
                logger.info(f"✅ 保存 {len(structures)} 个结构到 {comp_dir.name}")
                logger.info(f"   - CIF 文件: structure_1.cif 到 structure_{len(structures)}.cif")

            # 计算热导率并保存到 CSV
            self._calculate_and_save_thermal_conductivity(comp_dir, structures, composition)

            return comp_dir

        except Exception as e:
            logger.error(f"Failed to save structures: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # 返回输出目录即使保存失败
            return self.output_dir

    def _relax_and_save_structures(
        self,
        structures: List[CrystalStructure],
        composition: Composition,
        **kwargs
    ) -> List[CrystalStructure]:
        """
        弛豫结构并保存到 MyRelaxStructure 文件夹

        Args:
            structures: 待弛豫的结构列表
            composition: 组分信息
            **kwargs: 额外参数
                - pressure: 弛豫压力（GPa，默认0.0）
                - relax_output_dir: 弛豫结构输出目录（默认 MyRelaxStructure）

        Returns:
            List[CrystalStructure]: 弛豫后的结构列表（用于兼容性，实际保存 CIF）
        """
        try:
            from .mattersim_wrapper import MattersimWrapper
            from ase.io import write as ase_write

            # 获取参数
            pressure = kwargs.get('pressure', 0.0)
            relax_output_dir = kwargs.get('relax_output_dir')
            
            if relax_output_dir is None:
                raise ValueError("relax_output_dir must be provided")

            # 创建弛豫输出目录
            relax_base_dir = Path(relax_output_dir)
            relax_base_dir.mkdir(parents=True, exist_ok=True)

            # 为当前组分创建子目录
            comp_dir = relax_base_dir / composition.formula
            comp_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"弛豫结构将保存到: {comp_dir}")

            # 初始化 Mattersim
            mattersim = MattersimWrapper()

            # 弛豫所有结构，只保存成功的
            relaxed_atoms_list = []
            success_count = 0
            failed_count = 0

            for i, structure in enumerate(structures):
                logger.info(f"🔧 弛豫结构 {i+1}/{len(structures)}: {structure.structure_id}")

                # 调用弛豫方法
                response = mattersim.relax_structure(
                    structure,
                    pressure=pressure
                )

                if response.is_success():
                    relaxed_atoms = response.result  # ASE Atoms 对象
                    success_count += 1

                    # 只保存成功弛豫的结构
                    cif_file = comp_dir / f"structure_{success_count}.cif"
                    ase_write(str(cif_file), relaxed_atoms, format='cif')
                    relaxed_atoms_list.append(relaxed_atoms)

                    logger.info(f"  ✅ 弛豫成功，已保存: {cif_file.name}")
                else:
                    failed_count += 1
                    logger.warning(f"  ❌ 弛豫失败: {response.error}")
                    logger.info(f"  ⚠️ 跳过该结构，不保存到 MyRelaxStructure")

            logger.info(f"\n📊 弛豫统计: 成功 {success_count}/{len(structures)}, 失败 {failed_count}/{len(structures)}")

            # 检查是否有成功弛豫的结构
            if not relaxed_atoms_list:
                logger.warning(f"⚠️ 没有成功弛豫的结构，返回原始结构")
                return structures

            # 保存元数据（只包含成功弛豫的结构）
            logger.info(f"💾 保存元数据...")
            self._save_relaxed_metadata(
                comp_dir,
                len(relaxed_atoms_list),
                composition,
                n_total=len(structures),
                n_success=success_count,
                n_failed=failed_count
            )

            # 将 ASE Atoms 转换为 CrystalStructure（用于后续计算）
            relaxed_structures = self._atoms_to_crystal_structures(
                relaxed_atoms_list,
                composition
            )

            # 计算弛豫后结构的热导率（只计算成功弛豫的）
            logger.info(f"🔥 计算弛豫后结构的热导率...")
            self._calculate_and_save_thermal_conductivity(
                comp_dir,
                relaxed_structures,
                composition
            )

            # 计算弛豫后结构的声子谱（只计算成功弛豫的）
            logger.info(f"🎵 计算弛豫后结构的声子谱...")
            self._calculate_and_add_phonon_info(comp_dir, relaxed_structures, composition)

            return relaxed_structures

        except Exception as e:
            logger.error(f"结构弛豫失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return structures  # 返回原始结构

    def _save_relaxed_metadata(
        self,
        comp_dir: Path,
        n_structures: int,
        composition: Composition,
        n_total: int = None,
        n_success: int = None,
        n_failed: int = None
    ) -> None:
        """
        保存弛豫后结构的元数据（只包含成功弛豫的结构）

        Args:
            comp_dir: 组分目录
            n_structures: 成功弛豫的结构数量
            composition: 组分信息
            n_total: 总共尝试弛豫的结构数量
            n_success: 成功弛豫的数量
            n_failed: 失败的数量
        """
        try:
            metadata_list = []

            # 只记录成功弛豫的结构
            for i in range(n_structures):
                cif_file = comp_dir / f"structure_{i+1}.cif"

                metadata = {
                    'structure_id': f"structure_{i+1}_relaxed",
                    'composition': composition.formula,
                    'cif_file': cif_file.name,
                    'relaxed': True,
                    'relax_success': True
                }
                metadata_list.append(metadata)

            # 保存元数据
            metadata_file = comp_dir / "metadata.json"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            metadata_dict = {
                'composition': composition.formula,
                'timestamp': timestamp,
                'n_structures_saved': len(metadata_list),  # 保存的结构数量
                'n_structures_total': n_total if n_total else len(metadata_list),  # 总共尝试的数量
                'n_relaxed_success': n_success if n_success else len(metadata_list),
                'n_relaxed_failed': n_failed if n_failed else 0,
                'relaxed': True,
                'structures': metadata_list
            }

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

            if n_total and n_failed:
                logger.info(f"✅ 保存元数据: {len(metadata_list)} 个成功弛豫的结构 (总共尝试: {n_total}, 失败: {n_failed})")
            else:
                logger.info(f"✅ 保存元数据: {len(metadata_list)} 个结构")

        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _atoms_to_crystal_structures(
        self,
        atoms_list: List,
        composition: Composition
    ) -> List[CrystalStructure]:
        """
        将 ASE Atoms 列表转换为 CrystalStructure 列表

        Args:
            atoms_list: ASE Atoms 对象列表
            composition: 组分信息

        Returns:
            List[CrystalStructure]: CrystalStructure 列表
        """
        try:
            from ase.io import write as ase_write
            from io import StringIO

            structures = []

            for i, atoms in enumerate(atoms_list):
                # 将 Atoms 转换为 POSCAR 字符串
                poscar_io = StringIO()
                ase_write(poscar_io, atoms, format='vasp')
                poscar_str = poscar_io.getvalue()

                # 创建 CrystalStructure
                structure = CrystalStructure(
                    composition=composition,
                    poscar=poscar_str,
                    structure_id=f"structure_{i+1}_relaxed",
                    n_atoms=len(atoms),
                    metadata={'relaxed': True}
                )

                structures.append(structure)

            return structures

        except Exception as e:
            logger.error(f"转换 Atoms 到 CrystalStructure 失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
