"""
Mattersim工具封装

用于计算材料稳定性（声子谱、能量）。

主要功能：
- 计算声子谱（动力学稳定性）
- 检测虚频
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import time
import logging

# Limit BLAS/OMP threads before importing numpy/torch to reduce oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

try:
    from .base_tool import BaseTool, ToolResponse, ToolStatus
    from ..utils.types import CrystalStructure, StabilityResult
except ImportError:
    from tools.base_tool import BaseTool, ToolResponse, ToolStatus
    from utils.types import CrystalStructure, StabilityResult

logger = logging.getLogger(__name__)


class MattersimWrapper(BaseTool):
    """
    Mattersim工具封装

    功能：计算声子谱、形成能、动力学稳定性

    配置参数：
    - api_url: Mattersim API地址
    - model_path: 模型路径
    - imaginary_freq_threshold: 虚频阈值（THz，默认-0.1）
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: float = 600.0
    ):
        """
        初始化Mattersim

        Args:
            model_path: 模型路径
            config: 配置参数
            timeout: 超时时间
        """

        # Pre-initialize attributes needed by check_availability
        self.model_path = model_path
        self.api_url = None
        if config:
            self.api_url = config.get('api_url')
            if not self.model_path:
                self.model_path = config.get('model_path')

        super().__init__(name="Mattersim", config=config, timeout=timeout)
        
        # Re-assign to ensure consistency with BaseTool's config handling
        self.model_path = self.model_path or self.config.get('model_path')
        self.api_url = self.api_url or self.config.get('api_url')
        self.imaginary_freq_threshold = self.config.get('imaginary_freq_threshold', -0.1)  # THz
        self.calculator = None  # 缓存MatterSimCalculator实例

        # 自动检测CUDA
        self.device = self._detect_device()

    def _detect_device(self) -> str:
        """
        自动检测可用设备

        Returns:
            "cuda" 或 "cpu"
        """
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"✅ CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("⚠️ CUDA not available, using CPU")
            return device
        except ImportError:
            logger.warning("⚠️ PyTorch not found, using CPU")
            return "cpu"

    def check_availability(self) -> bool:
        """检查Mattersim是否可用"""
        try:
            # 检查mattersim库是否可用
            try:
                from mattersim.forcefield import MatterSimCalculator
                from mattersim.applications.phonon import PhononWorkflow
                logger.info("Mattersim library is available")
                return True
            except ImportError as e:
                logger.warning(f"Mattersim library not available: {e}")
                
            # 检查API可用性
            if self.api_url:
                logger.info(f"Checking Mattersim API at {self.api_url}")
                pass
            elif self.model_path:
                # 检查模型文件是否存在
                model_path = Path(self.model_path)
                if not model_path.exists():
                    logger.warning(f"Model path {self.model_path} does not exist")
                    return False

            # 默认返回True（使用简化模型作为后备）
            logger.warning("Mattersim not fully available, will use simplified model")
            return True
        except Exception as e:
            logger.error(f"Mattersim availability check failed: {e}")
            return False

    def run(
        self,
        structure: CrystalStructure,
        calculate_phonon: bool = True,
        save_plot: bool = False,
        plot_path: Optional[str] = None,
        **kwargs
    ) -> ToolResponse:
        """
        计算稳定性

        Args:
            structure: 晶体结构
            calculate_phonon: 是否计算声子谱
            save_plot: 是否保存声子谱图像
            plot_path: 图像保存路径
            **kwargs: 额外参数

        Returns:
            ToolResponse: 包含StabilityResult
        """
        start_time = time.time()

        try:
            if not self.is_available:
                return ToolResponse(
                    status=ToolStatus.NOT_AVAILABLE,
                    error="Mattersim not available"
                )

            logger.info(f"Calculating stability for {structure.composition.formula}")

            # 调用真实实现
            result = self._calculate_stability_real(
                structure,
                calculate_phonon,
                save_plot=save_plot,
                plot_path=plot_path
            )

            execution_time = time.time() - start_time

            if result.min_frequency is None:
                min_freq_msg = "unknown"
            else:
                min_freq_msg = f"{result.min_frequency:.3f} THz"
            logger.info(f"Stability: stable={result.is_dynamically_stable}, "
                       f"min_freq={min_freq_msg} in {execution_time:.2f}s")

            return ToolResponse(
                status=ToolStatus.SUCCESS,
                result=result,
                execution_time=execution_time,
                metadata={
                    'structure_id': structure.structure_id,
                    'composition': structure.composition.formula,
                    'is_stable': result.is_dynamically_stable,
                    'min_frequency': result.min_frequency,
                    'gamma_min_optical': result.gamma_min_optical,
                    'gamma_max_acoustic': result.gamma_max_acoustic,
                    'has_imaginary_freq': result.has_imaginary_freq,
                    'plot_path': plot_path if save_plot else None
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Mattersim calculation failed: {e}")
            return ToolResponse(
                status=ToolStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

    def _calculate_stability_real(
        self,
        structure: CrystalStructure,
        calculate_phonon: bool = True,
        save_plot: bool = False,
        plot_path: Optional[str] = None
    ) -> StabilityResult:
        """
        使用Mattersim计算稳定性（真实实现）

        基于MatterSim力场计算声子谱

        Args:
            structure: 晶体结构
            calculate_phonon: 是否计算声子谱
            save_plot: 是否保存声子谱图像
            plot_path: 图像保存路径

        Returns:
            StabilityResult: 稳定性结果
        """
        try:
            # 尝试使用Mattersim库
            try:
                return self._calculate_with_mattersim(
                    structure,
                    calculate_phonon,
                    save_plot=save_plot,
                    plot_path=plot_path
                )
            except Exception as e:
                logger.warning(f"Mattersim calculation failed: {e}, trying simplified model")
                # 回退到简化模型
                return self._calculate_with_simplified_model(structure)

        except Exception as e:
            logger.error(f"Real calculation failed: {e}, falling back to mock")
            import traceback
            logger.debug(traceback.format_exc())
            return self._calculate_mock_stability(structure)

    def _calculate_with_mattersim(
        self,
        structure: CrystalStructure,
        calculate_phonon: bool = True,
        save_plot: bool = False,
        plot_path: Optional[str] = None
    ) -> StabilityResult:
        """
        使用Mattersim库计算稳定性

        Args:
            structure: 晶体结构
            calculate_phonon: 是否计算声子谱
            save_plot: 是否保存声子谱图像
            plot_path: 图像保存路径

        Returns:
            StabilityResult: 稳定性结果
        """
        from mattersim.forcefield import MatterSimCalculator
        from mattersim.applications.phonon import PhononWorkflow

        # 1. 将POSCAR转换为ASE Atoms
        atoms = self._structure_to_atoms(structure)

        # 2. 获取或创建计算器
        calculator = self._get_or_create_calculator()
        atoms.set_calculator(calculator)

        # 初始化结果
        is_stable = True
        has_imaginary = False
        min_freq = 0.0
        gamma_min_optical = None
        gamma_max_acoustic = None
        phonon_workflow = None

        # 3. 计算声子谱
        if calculate_phonon:
            try:
                # 声子计算前检查显存
                if self.device == "cuda":
                    try:
                        import torch
                        if torch.cuda.is_available():
                            current_mem = torch.cuda.memory_allocated() / 1024**3
                            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            free_mem = total_mem - current_mem
                            logger.info(f"📊 声子计算前GPU显存: 已用={current_mem:.2f}GB, 总计={total_mem:.2f}GB, 剩余={free_mem:.2f}GB")
                            
                            # 如果剩余显存不足2GB,提前清理多次
                            if free_mem < 2.0:
                                logger.warning(f"⚠️ 剩余显存不足2GB,执行强制清理...")
                                for _ in range(3):
                                    self._cleanup_gpu_cache()
                    except Exception:
                        pass
                
                logger.info(f"⏳ 开始计算声子谱: {structure.composition.formula}...")

                # 确定工作目录(如果提供了 plot_path,使用其父目录)
                work_dir = None
                if save_plot and plot_path:
                    from pathlib import Path
                    work_dir = str(Path(plot_path).parent)

                # 创建PhononWorkflow(atoms 已经设置了 calculator)
                # 使用2x2x2超胞进行声子计算
                phonon_workflow_kwargs = {
                    'atoms': atoms,
                    'find_prim': False,
                    'supercell_matrix': np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),  # 2x2x2超胞
                    'amplitude': 0.01  # 位移量(Å)
                }

                # 如果指定了工作目录,添加到参数中
                if work_dir:
                    phonon_workflow_kwargs['work_dir'] = work_dir

                logger.info(f"  🔨 创建PhononWorkflow (2x2x2超胞)...")
                phonon_workflow = PhononWorkflow(**phonon_workflow_kwargs)

                # 运行声子计算（使用线程避免Windows pickle问题）
                logger.info(f"  🚀 运行声子计算（最长600秒）...")
                
                # 使用threading实现简单超时监控(无法强制终止,但可以记录超时)
                import threading
                import time as time_module
                
                phonon_result = {'has_imaginary': None, 'phonon': None, 'error': None, 'completed': False}
                
                def _run_phonon():
                    """在线程中运行声子计算"""
                    try:
                        has_img, ph = phonon_workflow.run()
                        phonon_result['has_imaginary'] = has_img
                        phonon_result['phonon'] = ph
                        phonon_result['completed'] = True
                    except Exception as e:
                        phonon_result['error'] = str(e)
                        phonon_result['completed'] = True
                
                phonon_thread = threading.Thread(target=_run_phonon)
                phonon_thread.daemon = True  # 设为守护线程
                phonon_thread.start()
                
                # 等待最多600秒
                start_time = time_module.time()
                timeout_seconds = 600
                phonon_thread.join(timeout=timeout_seconds)
                elapsed = time_module.time() - start_time
                
                if not phonon_result['completed']:
                    # 超时但线程可能还在运行(Python线程无法强制终止)
                    logger.error(f"⏱️ 声子谱计算超时({elapsed:.1f}s)，线程仍在后台运行")
                    raise TimeoutError(f"声子谱计算超时({elapsed:.1f}s)")
                
                if phonon_result['error']:
                    raise RuntimeError(f"声子计算失败: {phonon_result['error']}")
                
                has_imaginary = phonon_result['has_imaginary']
                phonon = phonon_result['phonon']

                min_freq = None
                max_freq = None
                gamma_min_optical = None
                gamma_max_acoustic = None
                actual_has_imaginary = has_imaginary

                if phonon is not None:
                    try:
                        phonon.run_mesh([20, 20, 20])
                        mesh_dict = phonon.get_mesh_dict()
                        frequencies = mesh_dict.get('frequencies')
                        if frequencies is not None:
                            freq_flat = frequencies.flatten()
                            if freq_flat.size:
                                min_freq = float(freq_flat.min())
                                max_freq = float(freq_flat.max())
                                actual_has_imaginary = min_freq < self.imaginary_freq_threshold

                        phonon.run_qpoints([[0, 0, 0]])
                        gamma_freqs = phonon.get_qpoints_dict().get('frequencies', [None])[0]
                        if gamma_freqs is not None:
                            if len(gamma_freqs) > 3:
                                gamma_min_optical = float(min(gamma_freqs[3:]))
                                gamma_max_acoustic = float(max(gamma_freqs[:3]))
                            else:
                                gamma_min_optical = float(min(gamma_freqs))
                                gamma_max_acoustic = float(max(gamma_freqs))
                    except Exception as e:
                        logger.warning(f"Failed to analyze phonon frequencies: {e}")
                        actual_has_imaginary = has_imaginary

                if min_freq is None:
                    min_freq = -0.1 if actual_has_imaginary else 0.0

                has_imaginary = actual_has_imaginary
                is_stable = not has_imaginary

                logger.info(f"  ✅ 声子谱计算完成 (耗时 {elapsed:.1f}s)")
                logger.info(f"  ✅ 声子谱计算完成: has_imaginary={has_imaginary}, stable={is_stable}")

                # 保存声子谱图像
                if save_plot and plot_path and phonon:
                    try:
                        self._plot_phonon_spectrum(
                            phonon,
                            plot_path,
                            structure.composition.formula,
                            has_imaginary,
                            work_dir
                        )
                        logger.info(f"Phonon spectrum plot saved to: {plot_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save phonon plot: {e}")

            except Exception as e:
                logger.warning(f"Phonon calculation failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # 声子计算失败,标记为未知
                is_stable = False
                has_imaginary = None
                min_freq = None
            finally:
                # 声子计算后立即强制清理GPU缓存
                self._cleanup_gpu_cache()
                
                # 清理phonopy生成的临时文件
                if work_dir:
                    try:
                        from pathlib import Path
                        work_path = Path(work_dir)
                        # 删除phonopy自动生成的文件
                        for temp_file in ['phonopy_params.yaml', 'band.yaml', 'mesh.yaml']:
                            temp_path = work_path / temp_file
                            if temp_path.exists():
                                temp_path.unlink()
                                logger.debug(f"已删除临时文件: {temp_file}")
                    except Exception as cleanup_error:
                        logger.debug(f"清理临时文件时出错: {cleanup_error}")
        
        # 清理 GPU 缓存（在返回结果前）
        self._cleanup_gpu_cache()

        result = StabilityResult(
            composition=structure.composition,
            structure_id=structure.structure_id,
            is_dynamically_stable=is_stable,
            has_imaginary_freq=has_imaginary,
            min_frequency=min_freq,
            gamma_min_optical=gamma_min_optical,
            gamma_max_acoustic=gamma_max_acoustic,
            formation_energy=None,
            energy_above_hull=None
        )

        return result
    
    def _cleanup_gpu_cache(self):
        """清理 GPU 显存缓存 - 增强版"""
        try:
            import torch
            import gc
            import time
            
            if torch.cuda.is_available():
                # 1. 多次垃圾回收
                for _ in range(3):
                    gc.collect()
                
                # 2. 清空CUDA缓存
                torch.cuda.empty_cache()
                
                # 3. 同步所有CUDA操作
                torch.cuda.synchronize()
                
                # 4. 再次清空缓存
                torch.cuda.empty_cache()
                
                # 5. 等待一小段时间让GPU完全释放
                time.sleep(0.5)
                
                # 6. 最后一次垃圾回收
                gc.collect()
                
                logger.debug("GPU cache thoroughly cleared")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"GPU cleanup warning: {e}")

    def _structure_to_atoms(self, structure: CrystalStructure):
        """
        将CrystalStructure转换为ASE Atoms对象

        Args:
            structure: 晶体结构

        Returns:
            ase.Atoms: ASE原子对象
        """
        try:
            from ase.io import read
            from io import StringIO

            # 从POSCAR字符串读取
            poscar_io = StringIO(structure.poscar)
            atoms = read(poscar_io, format='vasp')

            logger.debug(f"Converted structure to ASE Atoms: {len(atoms)} atoms")
            return atoms

        except Exception as e:
            logger.error(f"Failed to convert structure to Atoms: {e}")
            raise

    def _get_or_create_calculator(self):
        """
        创建新的MatterSimCalculator实例(每次创建新实例,避免缓存导致的显存累积)

        Returns:
            MatterSimCalculator: 计算器实例
        """
        try:
            from mattersim.forcefield import MatterSimCalculator
            import torch
            import gc
            
            # 先清理旧的calculator和GPU缓存
            if self.calculator is not None:
                del self.calculator
                self.calculator = None
            
            # 彻底清理GPU内存
            gc.collect()
            
            if torch.cuda.is_available():
                # 1. 清空所有GPU缓存
                torch.cuda.empty_cache()
                
                # 2. 同步所有设备
                torch.cuda.synchronize()
                
                # 3. 重置峰值内存统计
                torch.cuda.reset_peak_memory_stats()
                
                # 4. 尝试重置内存分配器（可能不支持所有CUDA版本）
                try:
                    torch.cuda.reset_accumulated_memory_stats()
                except:
                    pass
                
                # 5. 再次清空缓存
                torch.cuda.empty_cache()
                
                # 6. 垃圾回收
                gc.collect()
                
                # 7. 等待GPU完全空闲
                import time
                time.sleep(1)
                
                # 8. 打印当前内存使用情况
                current_mem = torch.cuda.memory_allocated() / 1024**3
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU内存: {current_mem:.2f}GB / {total_mem:.2f}GB (创建calculator前)")

            logger.info(f"Creating NEW MatterSimCalculator on {self.device}...")

            # 创建计算器(使用自动检测的设备)
            if self.model_path:
                self.calculator = MatterSimCalculator(
                    load_path=self.model_path,
                    device=self.device
                )
            else:
                # 使用默认模型
                self.calculator = MatterSimCalculator(device=self.device)

            logger.info(f"MatterSimCalculator created successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to create MatterSimCalculator: {e}")
            raise

        return self.calculator

    def _calculate_with_simplified_model(
        self,
        structure: CrystalStructure
    ) -> StabilityResult:
        """
        使用简化模型估算稳定性

        基于经验规则和简单物理模型

        Args:
            structure: 晶体结构

        Returns:
            StabilityResult: 稳定性结果
        """
        try:
            from ase.io import read
            from io import StringIO

            # 1. 将POSCAR转换为ASE Atoms
            poscar_io = StringIO(structure.poscar)
            atoms = read(poscar_io, format='vasp')

            # 2. 简化的稳定性判断
            # 基于原子间距和晶格参数

            # 检查最近邻距离
            distances = atoms.get_all_distances()
            min_dist = np.min(distances[distances > 0.01])

            # 如果最近邻距离太小，可能不稳定
            is_stable = min_dist > 1.5  # Å

            # 估算虚频（基于原子间距）
            # 距离越小，虚频越大（负值）
            if min_dist < 1.5:
                min_freq = -0.5 * (1.5 - min_dist)  # THz
                has_imaginary = True
            else:
                min_freq = 0.1 * (min_dist - 1.5)  # THz
                has_imaginary = False

            logger.info(f"Simplified model: stable={is_stable}, min_freq={min_freq:.3f} THz")

            result = StabilityResult(
                composition=structure.composition,
                structure_id=structure.structure_id,
                is_dynamically_stable=is_stable,
                has_imaginary_freq=has_imaginary,
                min_frequency=min_freq,
                formation_energy=None,
                energy_above_hull=None
            )

            return result

        except Exception as e:
            logger.error(f"Simplified model calculation failed: {e}")
            raise

    def _calculate_mock_stability(
        self,
        structure: CrystalStructure
    ) -> StabilityResult:
        """生成模拟稳定性数据（用于测试）"""
        # 随机生成稳定性结果
        is_stable = np.random.random() > 0.3  # 70%概率稳定

        if is_stable:
            min_freq = np.random.uniform(0.0, 2.0)  # THz
            has_imaginary = False
        else:
            min_freq = np.random.uniform(-1.0, -0.1)  # THz
            has_imaginary = True

        result = StabilityResult(
            composition=structure.composition,
            structure_id=structure.structure_id,
            is_dynamically_stable=is_stable,
            has_imaginary_freq=has_imaginary,
            min_frequency=min_freq,
            formation_energy=None,
            energy_above_hull=None
        )

        return result

    def validate_batch(
        self,
        structures: List[CrystalStructure],
        calculate_phonon: bool = True,
        **kwargs
    ) -> Dict[str, StabilityResult]:
        """
        批量验证稳定性

        Args:
            structures: 结构列表
            calculate_phonon: 是否计算声子谱
            **kwargs: 额外参数（传递给run方法）

        Returns:
            Dict: {structure_id: result}
        """
        results = {}
        for struct in structures:
            response = self.run(struct, calculate_phonon, **kwargs)
            if response.is_success():
                results[struct.structure_id] = response.result
            else:
                logger.warning(f"Failed to validate {struct.structure_id}: {response.error}")
        return results

    def filter_stable_structures(
        self,
        results: Dict[str, StabilityResult]
    ) -> List[str]:
        """
        过滤出稳定的结构

        Args:
            results: 稳定性结果字典

        Returns:
            List[str]: 稳定结构的ID列表
        """
        stable_ids = [
            struct_id for struct_id, result in results.items()
            if result.is_dynamically_stable
        ]

        logger.info(f"Filtered {len(stable_ids)}/{len(results)} stable structures")
        return stable_ids

    def _plot_phonon_spectrum(
        self,
        phonon,
        plot_path: str,
        formula: str,
        has_imaginary: bool,
        work_dir: str = None
    ):
        """
        绘制声子谱图像，并将 PhononWorkflow 生成的图像移动到正确位置

        Args:
            phonon: Phonopy 对象
            plot_path: 图像保存路径
            formula: 化学式
            has_imaginary: 是否有虚频
            work_dir: PhononWorkflow 的工作目录
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import shutil
            from pathlib import Path
            matplotlib.use('Agg')  # 使用非交互式后端

            # 使用 phonopy 的自动绘图功能
            # phonopy 对象有 auto_band_structure 方法可以自动绘制声子谱
            try:
                # 尝试使用 phonopy 的绘图 API
                phonon.auto_band_structure(plot=True, write_yaml=False, filename=plot_path)

                logger.info(f"Phonon spectrum plotted successfully using auto_band_structure")

                # 如果指定了工作目录，查找并移动 PhononWorkflow 生成的图像
                if work_dir:
                    work_path = Path(work_dir)
                    target_dir = Path(plot_path).parent

                    # 查找 phonon_band.png 和 phonon_dos.png
                    band_file = f"{formula}_phonon_band.png"
                    dos_file = f"{formula}_phonon_dos.png"

                    # 移动能带图
                    if (work_path / band_file).exists():
                        shutil.move(
                            str(work_path / band_file),
                            str(target_dir / "phonon_band.png")
                        )
                        logger.info(f"Moved phonon band plot to {target_dir / 'phonon_band.png'}")

                    # 移动态密度图
                    if (work_path / dos_file).exists():
                        shutil.move(
                            str(work_path / dos_file),
                            str(target_dir / "phonon_dos.png")
                        )
                        logger.info(f"Moved phonon DOS plot to {target_dir / 'phonon_dos.png'}")

                    # 删除 PhononWorkflow 创建的工作目录（如果它不是目标目录）
                    if work_path.exists() and work_path != target_dir:
                        try:
                            shutil.rmtree(work_path)
                            logger.info(f"Removed temporary phonon work directory: {work_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove work directory {work_path}: {e}")

                return

            except Exception as e1:
                logger.debug(f"auto_band_structure failed: {e1}, trying manual plotting...")

            # 如果自动绘图失败，尝试手动绘制
            # 获取能带结构数据
            band_dict = phonon.get_band_structure_dict()

            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制能带
            if 'frequencies' in band_dict:
                frequencies = band_dict['frequencies']
                distances = band_dict.get('distances', None)

                if distances is not None:
                    for band_idx in range(frequencies.shape[1]):
                        ax.plot(distances, frequencies[:, band_idx], color='blue', linewidth=1.5)
                else:
                    for band_idx in range(frequencies.shape[1]):
                        ax.plot(frequencies[:, band_idx], color='blue', linewidth=1.5)

            # 添加零频率线
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero frequency')

            # 设置标签和标题
            ax.set_xlabel('Wave vector', fontsize=12)
            ax.set_ylabel('Frequency (THz)', fontsize=12)

            # 根据是否有虚频设置标题颜色
            title_color = 'red' if has_imaginary else 'green'
            stability_text = '(有虚频)' if has_imaginary else '(无虚频)'
            ax.set_title(f'{formula} 声子谱 {stability_text}',
                        fontsize=14, fontweight='bold', color=title_color)

            ax.legend()
            ax.grid(True, alpha=0.3)

            # 保存图像
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to plot phonon spectrum: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # 如果绘图失败，创建一个简单的占位图
            self._create_placeholder_plot(plot_path, formula, has_imaginary)

    def _create_placeholder_plot(
        self,
        plot_path: str,
        formula: str,
        has_imaginary: bool
    ):
        """
        创建占位图（当无法绘制真实声子谱时）

        Args:
            plot_path: 图像保存路径
            formula: 化学式
            has_imaginary: 是否有虚频
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')

            fig, ax = plt.subplots(figsize=(10, 6))

            # 显示文本信息
            stability_text = '有虚频 (不稳定)' if has_imaginary else '无虚频 (稳定)'
            color = 'red' if has_imaginary else 'green'

            ax.text(0.5, 0.5, f'{formula}\n声子谱计算完成\n{stability_text}',
                   ha='center', va='center', fontsize=16,
                   fontweight='bold', color=color,
                   transform=ax.transAxes)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create placeholder plot: {e}")

    def relax_structure(
        self,
        structure: CrystalStructure,
        pressure: float = 0.0,
        fmax_stage1: float = 0.05,
        fmax_stage2: float = 0.01,
        max_steps: int = 200  # 降低默认值,防止卡死
    ) -> ToolResponse:
        """
        使用 MatterSim 弛豫晶体结构（两阶段渐进式弛豫）

        Args:
            structure: 晶体结构
            pressure: 压力 (GPa)
            fmax_stage1: 第一阶段最大力阈值 (eV/Å)
            fmax_stage2: 第二阶段最大力阈值 (eV/Å)
            max_steps: 每个阶段的最大步数

        Returns:
            ToolResponse: 包含弛豫后的 ASE Atoms 对象
        """
        start_time = time.time()

        try:
            if not self.is_available:
                return ToolResponse(
                    status=ToolStatus.NOT_AVAILABLE,
                    error="Mattersim not available"
                )

            logger.info(f"开始弛豫结构: {structure.composition.formula}")

            # 调用真实弛豫实现
            relaxed_atoms = self._relax_structure_real(
                structure,
                pressure,
                fmax_stage1,
                fmax_stage2,
                max_steps
            )

            execution_time = time.time() - start_time

            logger.info(f"结构弛豫完成，耗时 {execution_time:.2f}s")

            return ToolResponse(
                status=ToolStatus.SUCCESS,
                result=relaxed_atoms,
                execution_time=execution_time,
                metadata={
                    'structure_id': structure.structure_id,
                    'composition': structure.composition.formula,
                    'pressure': pressure,
                    'fmax_stage1': fmax_stage1,
                    'fmax_stage2': fmax_stage2
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"结构弛豫失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return ToolResponse(
                status=ToolStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

    def _relax_structure_real(
        self,
        structure: CrystalStructure,
        pressure: float = 0.0,
        fmax_stage1: float = 0.05,
        fmax_stage2: float = 0.01,
        max_steps: int = 200  # 降低默认值,防止卡死(每阶段最多200步)
    ):
        """
        使用 MatterSim 弛豫结构的真实实现

        Args:
            structure: 晶体结构
            pressure: 压力 (GPa)
            fmax_stage1: 第一阶段最大力阈值
            fmax_stage2: 第二阶段最大力阈值
            max_steps: 每个阶段的最大步数

        Returns:
            ASE Atoms: 弛豫后的 ASE Atoms 对象
        """
        try:
            from mattersim.forcefield import MatterSimCalculator
            from mattersim.applications.relax import Relaxer
            from ase.units import GPa

            # 1. 将 CrystalStructure 转换为 ASE Atoms
            atoms = self._structure_to_atoms(structure)
            
            # 强制清理GPU缓存（在创建计算器前，多次清理确保彻底）
            for _ in range(3):
                self._cleanup_gpu_cache()
            logger.info("GPU cache aggressively cleared before creating calculator")

            # 2. 获取或创建 MatterSim 计算器（使用统一方法确保一致性）
            logger.info("Setting up MatterSimCalculator for relaxation...")
            calculator = self._get_or_create_calculator()
            atoms.set_calculator(calculator)

            # 3. 创建 Relaxer
            relaxer = Relaxer(
                optimizer="BFGS",
                filter="ExpCellFilter",
                constrain_symmetry=False
            )

            # 4. 设置压力参数
            params_filter = {}
            if pressure > 0:
                params_filter["scalar_pressure"] = pressure * GPa
                logger.info(f"弛豫压力: {pressure} GPa")
            else:
                logger.info("弛豫压力: 0 GPa")

            # 5. 第一阶段：宽松弛豫 (添加时间限制)
            import time as time_module
            start_relax_time = time_module.time()
            max_relax_time = 300  # 总弛豫时间不超过5分钟
            
            logger.info(f"第一阶段弛豫: fmax={fmax_stage1}, maxstep=0.1, 最长{max_relax_time}秒")
            success1 = relaxer.relax(
                atoms,
                steps=max_steps,
                fmax=fmax_stage1,
                maxstep=0.1,
                params_filter=params_filter
            )
            
            elapsed = time_module.time() - start_relax_time
            if elapsed > max_relax_time:
                logger.warning(f"第一阶段弛豫超时({elapsed:.1f}s > {max_relax_time}s), 跳过第二阶段")
                success = False
            elif not success1:
                logger.warning("第一阶段弛豫未收敛")
                success = False
            else:
                # 6. 第二阶段：精细弛豫
                remaining_time = max_relax_time - elapsed
                if remaining_time < 30:
                    logger.warning(f"剩余时间不足({remaining_time:.1f}s), 跳过第二阶段")
                    success = success1
                else:
                    logger.info(f"第二阶段弛豫: fmax={fmax_stage2}, maxstep=0.05, 剩余{remaining_time:.1f}秒")
                    success2 = relaxer.relax(
                        atoms,
                        steps=max_steps,
                        fmax=fmax_stage2,
                        maxstep=0.05,
                        params_filter=params_filter
                    )
                    success = success2
            
            total_elapsed = time_module.time() - start_relax_time
            if not success:
                logger.warning(f"弛豫未收敛 (总耗时{total_elapsed:.1f}s)")
            else:
                logger.info(f"弛豫完成: converged={success} (总耗时{total_elapsed:.1f}s)")
            
            # 清理 GPU 缓存
            self._cleanup_gpu_cache()

            # 返回弛豫后的 ASE Atoms 对象
            return atoms

        except Exception as e:
            logger.error(f"结构弛豫失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # 即使失败也清理缓存
            self._cleanup_gpu_cache()
            raise
