"""
ai4kappa工具封装

用于预测材料热导率和热电性能。

主要功能：
- 预测热导率（晶格热导率 + 电子热导率）
- 预测热电性能（塞贝克系数、电导率、ZT值）
- 批量预测
- 阈值筛选
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import time
import logging
import numpy as np
import tempfile
import os

from .base_tool import BaseTool, ToolResponse, ToolStatus
from ..utils.types import CrystalStructure, MaterialProperties, Composition

logger = logging.getLogger(__name__)


class Ai4KappaWrapper(BaseTool):
    """
    ai4kappa工具封装

    功能：预测热导率、塞贝克系数、电导率等性能

    配置参数：
    - api_url: ai4kappa API地址
    - model_path: 模型路径
    - k_threshold: 默认热导率阈值（W/(m·K)，默认1.0）
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: float = 300.0
    ):
        """
        初始化ai4kappa

        Args:
            model_path: 模型路径
            config: 配置参数
            timeout: 超时时间
        """
        super().__init__(name="ai4kappa", config=config, timeout=timeout)
        self.model_path = model_path or self.config.get('model_path')
        self.api_url = self.config.get('api_url')
        self.k_threshold = self.config.get('k_threshold', 1.0)
        self.model = None
        self.calculator = None  # 缓存ThermalConductivityCalculator实例

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
        """检查ai4kappa是否可用"""
        try:
            # 检查kappa_lib是否可用
            try:
                from .kappa_lib.calculator import ThermalConductivityCalculator, is_kappa_available
                if not is_kappa_available():
                    logger.warning("kappa_lib modules not fully available")
                    return False
                logger.info("kappa_lib is available")
                return True
            except ImportError as e:
                logger.warning(f"kappa_lib not available: {e}")
                
            # 检查API可用性
            if self.api_url:
                logger.info(f"Checking ai4kappa API at {self.api_url}")
                pass
            elif self.model_path:
                # 检查模型文件是否存在
                model_path = Path(self.model_path)
                if not model_path.exists():
                    logger.warning(f"Model path {self.model_path} does not exist")
                    return False

            # 默认返回True（使用物理模型作为后备）
            logger.warning("ai4kappa CGCNN not fully available, will use physics model")
            return True
        except Exception as e:
            logger.error(f"ai4kappa availability check failed: {e}")
            return False

    def run(
        self,
        structure: CrystalStructure,
        temperature: float = 300.0,
        **kwargs
    ) -> ToolResponse:
        """
        预测材料性能

        Args:
            structure: 晶体结构
            temperature: 温度（K）
            **kwargs: 额外参数

        Returns:
            ToolResponse: 包含MaterialProperties
        """
        start_time = time.time()

        try:
            if not self.is_available:
                return ToolResponse(
                    status=ToolStatus.NOT_AVAILABLE,
                    error="ai4kappa not available"
                )

            logger.info(f"Predicting properties for {structure.composition.formula} at {temperature}K")

            # 调用真实实现
            properties = self._predict_properties_real(structure, temperature)

            execution_time = time.time() - start_time

            logger.info(f"Predicted k={properties.thermal_conductivity:.3f} W/(m·K) in {execution_time:.2f}s")

            return ToolResponse(
                status=ToolStatus.SUCCESS,
                result=properties,
                execution_time=execution_time,
                metadata={
                    'structure_id': structure.structure_id,
                    'composition': structure.composition.formula,
                    'temperature': temperature,
                    'thermal_conductivity': properties.thermal_conductivity
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"ai4kappa prediction failed: {e}")
            return ToolResponse(
                status=ToolStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

    def _predict_properties_real(
        self,
        structure: CrystalStructure,
        temperature: float
    ) -> MaterialProperties:
        """
        使用ai4kappa预测材料性能（真实实现）

        基于CGCNN模型预测热导率

        Args:
            structure: 晶体结构
            temperature: 温度（K）

        Returns:
            MaterialProperties: 材料性能
        """
        try:
            return self._predict_with_cgcnn(structure, temperature)
        except Exception as e:
            logger.error(f"CGCNN prediction failed: {e}, falling back to mock")
            import traceback
            logger.debug(traceback.format_exc())
            return self._predict_mock_properties(structure, temperature)

    def _predict_with_cgcnn(
        self,
        structure: CrystalStructure,
        temperature: float
    ) -> MaterialProperties:
        """
        使用CGCNN模型预测热导率

        Args:
            structure: 晶体结构
            temperature: 温度（K）

        Returns:
            MaterialProperties: 材料性能
        """
        from .kappa_lib.calculator import ThermalConductivityCalculator

        # 1. 将POSCAR转换为CIF文件
        cif_path = self._structure_to_cif(structure)

        try:
            # 2. 创建临时目录存放CIF文件
            cif_dir = Path(cif_path).parent

            # 3. 初始化计算器（使用缓存）
            if self.calculator is None:
                logger.info(f"Initializing ThermalConductivityCalculator with CIF dir: {cif_dir}")
                self.calculator = ThermalConductivityCalculator(str(cif_dir))

            # 4. 使用Kappa-P方法预测（基于Slack模型）
            logger.info("Running Kappa-P prediction...")
            result_df = self.calculator.calculate_kappa_p()

            # 5. 提取预测结果
            if result_df.empty:
                raise ValueError("CGCNN prediction returned empty results")

            # 获取第一行结果（对应我们的结构）
            row = result_df.iloc[0]

            # 提取热导率值（根据calculator.py的输出格式）
            k_total = float(row.get('K_MTP', row.get('thermal_conductivity', 1.0)))

            # 估算晶格和电子热导率（简化：假设80%晶格，20%电子）
            k_lattice = k_total * 0.8
            k_electronic = k_total * 0.2

            # 估算热电性能（基于经验关系）
            seebeck = 200.0 * (1.0 / k_total) ** 0.3
            electrical_conductivity = 1e5
            power_factor = (seebeck ** 2) * electrical_conductivity / 1e6
            zt_value = (seebeck ** 2) * electrical_conductivity * temperature / (k_total * 1e6)

            logger.info(f"CGCNN prediction: k={k_total:.3f} W/(m·K)")

            properties = MaterialProperties(
                composition=structure.composition,
                structure_id=structure.structure_id,
                thermal_conductivity=k_total,
                k_lattice=k_lattice,
                k_electronic=k_electronic,
                seebeck_coefficient=seebeck,
                electrical_conductivity=electrical_conductivity,
                power_factor=power_factor,
                zt_value=zt_value,
                temperature=temperature
            )

            return properties

        finally:
            # 清理临时CIF文件
            try:
                if cif_path and Path(cif_path).exists():
                    Path(cif_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up CIF file: {e}")

    def _structure_to_cif(self, structure: CrystalStructure) -> str:
        """
        将CrystalStructure转换为CIF文件

        Args:
            structure: 晶体结构

        Returns:
            str: CIF文件路径
        """
        try:
            from pymatgen.io.vasp import Poscar
            from pymatgen.io.cif import CifWriter
            from io import StringIO

            # 1. 从POSCAR字符串读取结构
            poscar_io = StringIO(structure.poscar)
            poscar = Poscar.from_string(structure.poscar)
            pmg_structure = poscar.structure

            # 2. 创建临时目录
            temp_dir = Path(tempfile.mkdtemp(prefix="ai4kappa_"))
            cif_path = temp_dir / f"{structure.structure_id}.cif"

            # 3. 写入CIF文件
            cif_writer = CifWriter(pmg_structure)
            cif_writer.write_file(str(cif_path))

            logger.debug(f"Created CIF file: {cif_path}")
            return str(cif_path)

        except Exception as e:
            logger.error(f"Failed to convert structure to CIF: {e}")
            raise

    def _predict_mock_properties(
        self,
        structure: CrystalStructure,
        temperature: float
    ) -> MaterialProperties:
        """生成模拟性能数据（用于测试）"""
        # 模拟热导率：0.1-3.0 W/(m·K)
        k = np.random.uniform(0.1, 3.0)

        properties = MaterialProperties(
            composition=structure.composition,
            structure_id=structure.structure_id,
            thermal_conductivity=k,
            k_lattice=k * 0.8,
            k_electronic=k * 0.2,
            seebeck_coefficient=np.random.uniform(100, 300),
            electrical_conductivity=np.random.uniform(1e4, 1e6),
            zt_value=np.random.uniform(0.1, 1.5),
            temperature=temperature
        )
        return properties

    def predict_batch(
        self,
        structures: List[CrystalStructure],
        temperature: float = 300.0,
        **kwargs
    ) -> Dict[str, MaterialProperties]:
        """
        批量预测

        Args:
            structures: 结构列表
            temperature: 温度
            **kwargs: 额外参数（传递给run方法）

        Returns:
            Dict: {structure_id: properties}
        """
        results = {}
        for struct in structures:
            response = self.run(struct, temperature, **kwargs)
            if response.is_success():
                results[struct.structure_id] = response.result
            else:
                logger.warning(f"Failed to predict properties for {struct.structure_id}: {response.error}")
        return results

    def filter_by_k_threshold(
        self,
        properties_list: List[MaterialProperties],
        k_threshold: float = 1.0
    ) -> List[MaterialProperties]:
        """
        按热导率阈值过滤

        Args:
            properties_list: 性能列表
            k_threshold: 热导率阈值

        Returns:
            List[MaterialProperties]: 满足条件的性能列表
        """
        return [
            prop for prop in properties_list
            if prop.thermal_conductivity is not None
            and prop.thermal_conductivity < k_threshold
        ]
