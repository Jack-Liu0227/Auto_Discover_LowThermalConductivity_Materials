"""
外部工具基类定义

定义所有外部工具的统一接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import time


class ToolStatus(Enum):
    """工具状态"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NOT_AVAILABLE = "not_available"


@dataclass
class ToolResponse:
    """工具响应"""
    status: ToolStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def is_success(self) -> bool:
        """是否成功"""
        return self.status == ToolStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


class BaseTool(ABC):
    """
    外部工具基类
    
    所有外部工具封装必须继承此类。
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        timeout: float = 300.0
    ):
        """
        初始化工具
        
        Args:
            name: 工具名称
            config: 配置参数
            timeout: 超时时间（秒）
        """
        self.name = name
        self.config = config or {}
        self.timeout = timeout
        self.is_available = self.check_availability()
    
    @abstractmethod
    def check_availability(self) -> bool:
        """
        检查工具是否可用
        
        Returns:
            bool: 是否可用
        """
        pass
    
    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> ToolResponse:
        """
        运行工具
        
        Args:
            input_data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            ToolResponse: 工具响应
        """
        pass
    
    def run_with_retry(
        self,
        input_data: Any,
        max_retries: int = 3,
        **kwargs
    ) -> ToolResponse:
        """
        带重试的运行
        
        Args:
            input_data: 输入数据
            max_retries: 最大重试次数
            **kwargs: 额外参数
            
        Returns:
            ToolResponse: 工具响应
        """
        for attempt in range(max_retries):
            response = self.run(input_data, **kwargs)
            if response.is_success():
                return response
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
        
        return response
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', available={self.is_available})"

