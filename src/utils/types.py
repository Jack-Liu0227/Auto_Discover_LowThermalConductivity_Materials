"""
核心数据结构和类型定义

定义整个系统使用的核心数据结构，确保类型一致性和接口规范。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import numpy as np


# ============================================================================
# 枚举类型
# ============================================================================

class AgentStatus(Enum):
    """Agent状态"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class StabilityStatus(Enum):
    """稳定性状态"""
    STABLE = "stable"
    UNSTABLE = "unstable"
    UNKNOWN = "unknown"


class DecisionType(Enum):
    """决策类型"""
    PASS = "pass"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


# ============================================================================
# 组分相关数据结构
# ============================================================================

@dataclass
class Composition:
    """材料组分"""
    formula: str  # 化学式，如 "Bi2Te3"
    elements: Dict[str, float]  # 元素组成，如 {"Bi": 2, "Te": 3}
    normalized_formula: Optional[str] = None  # 归一化化学式
    
    def __post_init__(self):
        """初始化后处理"""
        if self.normalized_formula is None:
            self.normalized_formula = self.formula
    
    def __hash__(self):
        return hash(self.formula)
    
    def __eq__(self, other):
        if not isinstance(other, Composition):
            return False
        return self.formula == other.formula


@dataclass
class CompositionFeatures:
    """组分特征"""
    formula: str
    features: np.ndarray  # 特征向量 (30-50维)
    feature_names: List[str]  # 特征名称
    
    # 元素组成特征
    n_elements: int = 0
    element_fractions: Dict[str, float] = field(default_factory=dict)
    
    # 物理化学特征
    avg_atomic_mass: float = 0.0
    mass_contrast: float = 0.0  # Γ_M
    electronegativity_diff: float = 0.0
    avg_atomic_radius: float = 0.0
    radius_mismatch: float = 0.0
    
    # 领域知识特征
    lone_pair_ratio: float = 0.0  # R_lone
    has_heavy_elements: bool = False
    has_chalcogenides: bool = False
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return self.features
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'formula': self.formula,
            'features': self.features.tolist(),
            'feature_names': self.feature_names,
            'n_elements': self.n_elements,
            'avg_atomic_mass': self.avg_atomic_mass,
            'mass_contrast': self.mass_contrast,
            'lone_pair_ratio': self.lone_pair_ratio
        }


@dataclass
class CandidateComposition:
    """候选组分（带评分）"""
    composition: Composition
    score: float  # 综合评分 (0-1)
    predicted_k: Optional[float] = None  # 预测热导率
    probability: Optional[float] = None  # P(k < 1.0)
    novelty: Optional[float] = None  # 新颖性评分
    diversity: Optional[float] = None  # 多样性评分
    features: Optional[CompositionFeatures] = None
    reasoning: str = ""  # 推理依据
    source: str = "unknown"  # 来源：exploitation/exploration
    
    def __lt__(self, other):
        """用于排序"""
        return self.score < other.score


# ============================================================================
# 结构相关数据结构
# ============================================================================

@dataclass
class CrystalStructure:
    """晶体结构"""
    composition: Composition
    structure_id: str  # 结构唯一标识
    poscar: str  # POSCAR格式字符串
    space_group: Optional[str] = None
    lattice_params: Optional[Dict[str, float]] = None  # a, b, c, alpha, beta, gamma
    n_atoms: int = 0
    quality_score: float = 0.0  # 结构质量评分
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据（如弛豫信息）

    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'composition': self.composition.formula,
            'structure_id': self.structure_id,
            'space_group': self.space_group,
            'lattice_params': self.lattice_params,
            'n_atoms': self.n_atoms,
            'quality_score': self.quality_score,
            'metadata': self.metadata
        }


# ============================================================================
# 性能预测相关数据结构
# ============================================================================

@dataclass
class MaterialProperties:
    """材料性能"""
    composition: Composition
    structure_id: str
    
    # 热导率
    thermal_conductivity: Optional[float] = None  # k (W/(m·K))
    k_lattice: Optional[float] = None  # 晶格热导率
    k_electronic: Optional[float] = None  # 电子热导率
    
    # 热电性能
    seebeck_coefficient: Optional[float] = None  # S (μV/K)
    electrical_conductivity: Optional[float] = None  # σ (S/m)
    power_factor: Optional[float] = None  # PF (μW/(cm·K²))
    zt_value: Optional[float] = None  # ZT
    
    # 测量条件
    temperature: Optional[float] = None  # K
    
    def meets_target(self, k_threshold: float = 1.0) -> bool:
        """是否满足目标"""
        if self.thermal_conductivity is None:
            return False
        return self.thermal_conductivity < k_threshold

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'composition': self.composition.formula,
            'structure_id': self.structure_id,
            'thermal_conductivity': self.thermal_conductivity,
            'seebeck_coefficient': self.seebeck_coefficient,
            'zt_value': self.zt_value,
            'temperature': self.temperature
        }


# ============================================================================
# 稳定性相关数据结构
# ============================================================================

@dataclass
class StabilityResult:
    """稳定性验证结果"""
    composition: Composition
    structure_id: str

    # 动力学稳定性
    is_dynamically_stable: bool = False
    has_imaginary_freq: bool = True
    min_frequency: Optional[float] = None  # THz
    gamma_min_optical: Optional[float] = None  # THz
    gamma_max_acoustic: Optional[float] = None  # THz

    # 热力学稳定性
    formation_energy: Optional[float] = None  # eV/atom
    energy_above_hull: Optional[float] = None  # eV/atom

    # 机械稳定性
    is_mechanically_stable: Optional[bool] = None

    # 综合判定
    status: StabilityStatus = StabilityStatus.UNKNOWN
    confidence: float = 0.0

    def is_stable(self) -> bool:
        """是否稳定"""
        return self.status == StabilityStatus.STABLE

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'composition': self.composition.formula,
            'structure_id': self.structure_id,
            'is_dynamically_stable': self.is_dynamically_stable,
            'formation_energy': self.formation_energy,
            'status': self.status.value,
            'confidence': self.confidence
        }


# ============================================================================
# Loop相关数据结构
# ============================================================================

@dataclass
class LoopIteration:
    """单次Loop迭代结果"""
    iteration: int

    # 输入
    n_candidates: int
    exploitation_ratio: float

    # 中间结果
    candidates: List[CandidateComposition] = field(default_factory=list)
    structures: List[CrystalStructure] = field(default_factory=list)
    properties: List[MaterialProperties] = field(default_factory=list)
    stability_results: List[StabilityResult] = field(default_factory=list)

    # 输出
    success_materials: List[Dict[str, Any]] = field(default_factory=list)
    failure_materials: List[Dict[str, Any]] = field(default_factory=list)

    # 统计
    success_rate: float = 0.0
    novelty_rate: float = 0.0
    avg_k: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'iteration': self.iteration,
            'n_candidates': self.n_candidates,
            'exploitation_ratio': self.exploitation_ratio,
            'n_success': len(self.success_materials),
            'n_failure': len(self.failure_materials),
            'success_rate': self.success_rate,
            'novelty_rate': self.novelty_rate,
            'avg_k': self.avg_k
        }


@dataclass
class LoopState:
    """Loop状态"""
    current_iteration: int = 0
    max_iterations: int = 100

    # 累积结果
    total_success: int = 0
    total_evaluated: int = 0
    success_materials: List[Dict[str, Any]] = field(default_factory=list)

    # 历史记录
    iterations: List[LoopIteration] = field(default_factory=list)
    success_rate_history: List[float] = field(default_factory=list)

    # 终止条件
    target_materials: int = 20
    patience: int = 10
    consecutive_no_discovery: int = 0

    def should_terminate(self) -> bool:
        """是否应该终止"""
        # 条件1: 达到最大迭代次数
        if self.current_iteration >= self.max_iterations:
            return True

        # 条件2: 发现足够材料
        if self.total_success >= self.target_materials:
            return True

        # 条件3: 连续无新发现
        if self.consecutive_no_discovery >= self.patience:
            return True

        # 条件4: 成功率过低
        if len(self.success_rate_history) >= 10:
            recent_rates = self.success_rate_history[-10:]
            if all(r < 0.05 for r in recent_rates):
                return True

        return False

    def update(self, iteration_result: LoopIteration):
        """更新状态"""
        self.iterations.append(iteration_result)
        self.success_rate_history.append(iteration_result.success_rate)
        self.total_success += len(iteration_result.success_materials)
        self.total_evaluated += iteration_result.n_candidates
        self.success_materials.extend(iteration_result.success_materials)

        # 更新连续无发现计数
        if len(iteration_result.success_materials) == 0:
            self.consecutive_no_discovery += 1
        else:
            self.consecutive_no_discovery = 0


# ============================================================================
# Agent响应数据结构
# ============================================================================

@dataclass
class AgentResponse:
    """Agent响应"""
    status: AgentStatus
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0  # 秒
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """是否成功"""
        return self.status == AgentStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'status': self.status.value,
            'data': self.data,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }
