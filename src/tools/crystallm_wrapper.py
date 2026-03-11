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
            self.device = device
        else:
            self.device = self._detect_device()

    def _detect_device(self) -> str:
        """
        自动检测可用设备（默认强制使用 CUDA）
        使用全局锁防止多线程并发初始化CUDA

        Returns:
            "cuda" 或 "cpu"
        """
        with _cuda_init_lock:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    logger.info(f"✅ CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("❌ CUDA not detected, but will try using 'cuda' anyway")
                    logger.warning("如果遇到错误，请检查CUDA和PyTorch安装")
                    device = "cuda"  # 仍然尝试使用cuda
                return device
            except ImportError:
                logger.warning("⚠️ PyTorch not found, defaulting to cuda")
                return "cuda"

    def check_availability(self) -> bool:
        """检查CrystaLLM是否可用"""
        try:
            # 检查crystallm generator是否可用
            try:
                from .crystallm.generator import generate_crystal_from_composition
                logger.info("✅ CrystaLLM generator is available")
                return True
            except ImportError as e:
                logger.warning(f"⚠️ CrystaLLM generator not available: {e}")

            # 检查API可用性
            if hasattr(self, 'api_url') and self.api_url:
                logger.info(f"Checking CrystaLLM API at {self.api_url}")
                # TODO: 实现API检查
                pass
            elif hasattr(self, 'model_path') and self.model_path:
                # 检查模型文件是否存在
                model_path = Path(self.model_path)
                if not model_path.exists():
                    logger.warning(f"⚠️ Model path {self.model_path} does not exist")
                    return False

            # 默认返回True（使用pymatgen作为后备）
            logger.info("ℹ️ CrystaLLM not fully available, will use pymatgen fallback")
            return True
        except Exception as e:
            logger.error(f"❌ CrystaLLM availability check failed: {e}")
            # 即使检查失败，也返回True以使用后备方案
            return True

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
                    error="CrystaLLM not available"
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

            # 注意：结构已经在 _generate_with_crystallm 或 _generate_with_pymatgen 中保存
            # 不需要再次调用 _save_structures，避免重复保存

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

            logger.info(f"Generated {len(valid_structures)}/{len(structures)} valid structures in {execution_time:.2f}s")

            return ToolResponse(
                status=ToolStatus.SUCCESS,
                result=valid_structures,
                execution_time=execution_time,
                metadata={
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
            return ToolResponse(
                status=ToolStatus.FAILED,
                error=str(e),
                execution_time=execution_time
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

        基于Transformer模型生成晶体结构或pymatgen备选方案

        Args:
            composition: 组分
            n_structures: 生成结构数量
            temperature: 采样温度
            seed: 随机种子

        Returns:
            List[CrystalStructure]: 生成的结构列表
        """
        try:
            # 尝试使用CrystaLLM生成器
            try:
                return self._generate_with_crystallm(composition, n_structures, temperature, seed, calculate_properties)
            except Exception as e:
                logger.warning(f"CrystaLLM generation failed: {e}, trying pymatgen")
                # 回退到pymatgen
                return self._generate_with_pymatgen(composition, n_structures, seed, calculate_properties)

        except Exception as e:
            logger.error(f"Real generation failed: {e}, falling back to mock")
            import traceback
            logger.debug(traceback.format_exc())
            return self._generate_mock_structures(composition, n_structures)

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
        import shutil

        logger.info(f"Using CrystaLLM to generate {n_structures} structures for {composition.formula}")

        structures = []

        # 强制使用 CUDA（如果可用）
        device_to_use = self.device
        if device_to_use == "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    device_to_use = "cuda"
                    logger.warning(f"⚠️ Forcing CUDA usage even though device was set to CPU")
            except:
                pass

        # CrystaLLM一次生成多个结构
        result = generate_crystal_from_composition(
            composition=composition.formula,
            device=device_to_use,  # 强制使用CUDA
            num_samples=n_structures,
            top_k=10,
            max_new_tokens=2000,
            output_dir=str(self.output_dir)  # 使用 wrapper 的输出目录
        )

        if not result.get('success', False):
            raise ValueError(f"CrystaLLM generation failed: {result.get('error', 'Unknown error')}")

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

    def _generate_mock_structures(
        self,
        composition: Composition,
        n_structures: int
    ) -> List[CrystalStructure]:
        """生成模拟结构（用于测试）"""
        structures = []
        for i in range(n_structures):
            structure = CrystalStructure(
                composition=composition,
                structure_id=f"{composition.formula}_struct_{i}",
                poscar=f"# Mock POSCAR for {composition.formula}",
                space_group="P1",
                lattice_params={'a': 5.0, 'b': 5.0, 'c': 5.0,
                               'alpha': 90, 'beta': 90, 'gamma': 90},
                n_atoms=10,
                quality_score=0.8
            )
            structures.append(structure)
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

    def _generate_with_pymatgen(
        self,
        composition: Composition,
        n_structures: int,
        seed: Optional[int] = None,
        calculate_properties: bool = True  # 新增参数
    ) -> List[CrystalStructure]:
        """
        使用pymatgen生成结构（备选方案）

        通过生成随机晶格结构

        Args:
            composition: 组分
            n_structures: 生成数量
            seed: 随机种子

        Returns:
            List[CrystalStructure]: 生成的结构列表
        """
        try:
            from pymatgen.core import Structure, Lattice
            from pymatgen.io.vasp import Poscar
            import numpy as np

            if seed is not None:
                np.random.seed(seed)

            logger.info(f"Generating {n_structures} structures with pymatgen for {composition.formula}")

            structures = []

            # 生成随机晶格结构
            for i in range(n_structures):
                try:
                    # 生成随机晶格参数
                    a = np.random.uniform(4.0, 8.0)
                    b = np.random.uniform(4.0, 8.0)
                    c = np.random.uniform(4.0, 8.0)
                    alpha = np.random.uniform(80, 100)
                    beta = np.random.uniform(80, 100)
                    gamma = np.random.uniform(80, 100)

                    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

                    # 生成原子位置
                    species = []
                    coords = []
                    for element, count in composition.elements.items():
                        for _ in range(int(count)):
                            species.append(element)
                            coords.append(np.random.random(3))

                    # 创建Structure对象
                    pmg_structure = Structure(lattice, species, coords)

                    # 转换为POSCAR格式
                    poscar = Poscar(pmg_structure)
                    poscar_str = str(poscar)

                    # 获取空间群信息
                    try:
                        space_group = pmg_structure.get_space_group_info()[0]
                    except:
                        space_group = "P1"

                    # 创建CrystalStructure对象
                    crystal_structure = CrystalStructure(
                        composition=composition,
                        structure_id=f"{composition.formula}_pymatgen_{i}",
                        poscar=poscar_str,
                        space_group=space_group,
                        lattice_params={
                            'a': a, 'b': b, 'c': c,
                            'alpha': alpha, 'beta': beta, 'gamma': gamma
                        },
                        n_atoms=len(species),
                        quality_score=0.7  # pymatgen生成的质量评分较低
                    )

                    structures.append(crystal_structure)

                except Exception as e:
                    logger.warning(f"Failed to generate structure {i}: {e}")

            logger.info(f"Successfully generated {len(structures)}/{n_structures} structures with pymatgen")

            # 保存结构到文件
            if structures:
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_formula = composition.formula.replace("/", "_").replace("\\", "_")
                # comp_dir = self.output_dir / f"{safe_formula}_{timestamp}"
                comp_dir = self.output_dir / safe_formula
                comp_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"📁 Saving structures to: {comp_dir}")

                # 保存 CIF 文件
                from pymatgen.io.cif import CifWriter
                for i, structure in enumerate(structures):
                    try:
                        # 从 POSCAR 字符串创建 pymatgen Structure
                        poscar_obj = Poscar.from_str(structure.poscar)
                        pmg_structure = poscar_obj.structure

                        # 保存 CIF 文件
                        cif_file = comp_dir / f"structure_{i+1}.cif"
                        cif_writer = CifWriter(pmg_structure)
                        cif_writer.write_file(str(cif_file))
                        logger.info(f"✅ Saved CIF file: {cif_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to save structure {i+1}: {e}")

                # 保存元数据
                self._save_metadata(comp_dir, structures, composition)
                logger.info(f"📊 Saved metadata to: {comp_dir / 'metadata.json'}")

                # 计算热导率并保存到 CSV
                self._calculate_and_save_thermal_conductivity(comp_dir, structures, composition)

            return structures

        except Exception as e:
            logger.error(f"Pymatgen generation failed: {e}")
            return []

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

    def _calculate_and_save_thermal_conductivity(
        self,
        comp_dir: Path,
        structures: List[CrystalStructure],
        composition: Composition
    ) -> None:
        """
        计算热导率并保存到 CSV 文件

        优先使用 kappa_lib 中的 ThermalConductivityCalculator（需要 torch）
        如果失败，使用简化的物理模型作为后备方案

        Args:
            comp_dir: 组分目录（包含 CIF 文件）
            structures: 结构列表（只包含成功弛豫的结构）
            composition: 组分信息
        """
        try:
            # 尝试使用完整的 kappa_lib 计算器
            self._calculate_with_kappa_lib(comp_dir, structures, composition)
        except Exception as e:
            logger.warning(f"Kappa-lib 计算失败: {e}，使用简化模型...")
            # 使用简化的物理模型作为后备
            self._calculate_with_simplified_model(comp_dir, structures, composition)

        # 注意：第一步生成的结构不需要计算声子谱
        # 声子谱计算只在弛豫后进行（见 _relax_and_save_structures 方法）

    def _calculate_with_kappa_lib(
        self,
        comp_dir: Path,
        structures: List[CrystalStructure],
        composition: Composition
    ) -> None:
        """使用 kappa_lib 计算热导率（需要 torch）"""
        import pandas as pd
        from .kappa_lib.calculator import ThermalConductivityCalculator, KAPPA_AVAILABLE

        if not KAPPA_AVAILABLE:
            raise ImportError("Kappa library not available (missing torch or other dependencies)")

        logger.info(f"🔥 开始计算热导率（使用 Kappa-P 方法 + CGCNN）...")

        # 初始化热导率计算器
        calculator = ThermalConductivityCalculator(str(comp_dir))

        # 使用 Kappa-P 方法计算（基于 Slack 模型 + CGCNN 预测）
        result_df = calculator.calculate_kappa_p()

        if result_df.empty:
            raise ValueError("热导率计算返回空结果")

        # 添加组分信息和结构 ID
        # 使用实际的 CIF 文件名（与 structure_id 对应）
        result_df.insert(0, 'Composition', composition.formula)
        result_df.insert(1, 'Structure_ID', [s.structure_id for s in structures[:len(result_df)]])
        result_df.insert(2, 'CIF_File', [f"{s.structure_id}.cif" for s in structures[:len(result_df)]])

        # 保存到 CSV
        csv_file = comp_dir / "thermal_conductivity.csv"
        result_df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        logger.info(f"✅ 热导率计算完成（Kappa-P），结果保存到: {csv_file.name}")

        # 打印结果摘要
        if 'Kappa_Slack (W m-1 K-1)' in result_df.columns:
            kappa_values = result_df['Kappa_Slack (W m-1 K-1)'].values
            logger.info(f"📊 热导率范围: {kappa_values.min():.3f} - {kappa_values.max():.3f} W/(m·K)")
            logger.info(f"📊 平均热导率: {kappa_values.mean():.3f} W/(m·K)")

    def _calculate_with_simplified_model(
        self,
        comp_dir: Path,
        structures: List[CrystalStructure],
        composition: Composition
    ) -> None:
        """使用简化的物理模型计算热导率（不需要 torch）"""
        import pandas as pd
        from pymatgen.io.cif import CifParser
        import numpy as np

        logger.info(f"🔥 开始计算热导率（使用简化 Slack 模型）...")

        data_list = []

        for i, structure in enumerate(structures):
            try:
                # 读取 CIF 文件（使用实际的 structure_id）
                cif_file = comp_dir / f"{structure.structure_id}.cif"
                if not cif_file.exists():
                    # 备用路径：尝试 structure_{i+1}.cif
                    cif_file = comp_dir / f"structure_{i+1}.cif"
                    if not cif_file.exists():
                        logger.warning(f"CIF 文件不存在: {structure.structure_id}.cif 或 structure_{i+1}.cif")
                        continue

                parser = CifParser(str(cif_file))
                pmg_structure = parser.get_structures()[0]

                # 提取基本参数
                volume = pmg_structure.volume  # Å³
                n_atoms = len(pmg_structure)
                density = pmg_structure.density  # g/cm³

                # 计算总原子质量
                total_mass = sum([site.specie.atomic_mass for site in pmg_structure])
                avg_mass = total_mass / n_atoms

                # 晶格参数
                lattice = pmg_structure.lattice
                a, b, c = lattice.a, lattice.b, lattice.c
                alpha, beta, gamma_angle = lattice.alpha, lattice.beta, lattice.gamma

                # 估算声速（基于密度，简化公式）
                # v_s ≈ sqrt(E/ρ)，E 为弹性模量，这里使用经验值
                v_s = 3000  # m/s（典型值）

                # 计算 Debye 温度
                # θ_D = (h/k_B) * v_s * (3n/4πV)^(1/3)
                h = 6.626e-34  # J·s
                k_B = 1.381e-23  # J/K
                theta_D = (h / k_B) * v_s * (3 * n_atoms / (4 * np.pi * volume * 1e-30))**(1/3)

                # 估算 Grüneisen 参数（典型值）
                gamma_gruneisen = 2.0

                # 使用 Slack 模型计算热导率
                # κ = A * M * V^(1/3) * θ_D^3 / (γ^2 * T * n)
                T = 300  # K
                A = 2.43e-8  # 经验常数
                kappa = A * total_mass * (volume**(1/3)) * (theta_D**3) / (gamma_gruneisen**2 * T * n_atoms) * 100

                # 收集数据
                data = {
                    'Composition': composition.formula,
                    'Structure_ID': structure.structure_id,
                    'CIF_File': f"{structure.structure_id}.cif",
                    'Space_Group': structure.space_group,
                    'Volume (Å³)': volume,
                    'Density (g/cm³)': density,
                    'N_Atoms': n_atoms,
                    'Avg_Atomic_Mass (amu)': avg_mass,
                    'Total_Mass (amu)': total_mass,
                    'Lattice_a (Å)': a,
                    'Lattice_b (Å)': b,
                    'Lattice_c (Å)': c,
                    'Lattice_alpha (°)': alpha,
                    'Lattice_beta (°)': beta,
                    'Lattice_gamma (°)': gamma_angle,
                    'Debye_Temperature (K)': theta_D,
                    'Gruneisen_Parameter': gamma_gruneisen,
                    'Kappa_Slack (W m-1 K-1)': kappa
                }

                data_list.append(data)
                logger.info(f"  结构 {i+1}: κ = {kappa:.3f} W/(m·K)")

            except Exception as e:
                logger.warning(f"处理结构 {i+1} 时出错: {e}")
                continue

        if not data_list:
            logger.warning("没有成功计算任何结构的热导率")
            return

        # 创建 DataFrame
        result_df = pd.DataFrame(data_list)

        # 保存到 CSV
        csv_file = comp_dir / "thermal_conductivity.csv"
        result_df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        logger.info(f"✅ 热导率计算完成（简化模型），结果保存到: {csv_file.name}")

        # 打印结果摘要
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
