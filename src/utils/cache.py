"""
缓存机制

提供LRU缓存功能，用于缓存：
- 特征提取结果
- 模型预测结果
- CrystaLLM生成结果
"""

import logging
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps
from collections import OrderedDict
import json

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU (Least Recently Used) 缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        
        logger.info(f"LRUCache initialized with max_size={max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在则返回None
        """
        if key in self.cache:
            # 移到末尾（最近使用）
            self.cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"Cache hit for key: {key[:50]}...")
            return self.cache[key]
        else:
            self.misses += 1
            logger.debug(f"Cache miss for key: {key[:50]}...")
            return None
    
    def put(self, key: str, value: Any):
        """
        存入缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        if key in self.cache:
            # 更新并移到末尾
            self.cache.move_to_end(key)
        else:
            # 新增
            if len(self.cache) >= self.max_size:
                # 删除最久未使用的（第一个）
                removed_key = next(iter(self.cache))
                del self.cache[removed_key]
                logger.debug(f"Cache full, removed key: {removed_key[:50]}...")
        
        self.cache[key] = value
        logger.debug(f"Cache put for key: {key[:50]}...")
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class PersistentCache:
    """持久化缓存（基于文件系统）"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        初始化持久化缓存
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PersistentCache initialized with cache_dir={cache_dir}")
    
    def _get_cache_path(self, key: str) -> Path:
        """
        获取缓存文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            缓存文件路径
        """
        # 使用MD5哈希作为文件名
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在则返回None
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            logger.debug(f"Cache miss for key: {key[:50]}...")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            logger.debug(f"Cache hit for key: {key[:50]}...")
            return value
        except Exception as e:
            logger.error(f"Failed to load cache for key {key[:50]}...: {e}")
            return None
    
    def put(self, key: str, value: Any):
        """
        存入缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Cache put for key: {key[:50]}...")
        except Exception as e:
            logger.error(f"Failed to save cache for key {key[:50]}...: {e}")

    def clear(self):
        """清空缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Failed to delete cache file {cache_file}: {e}")

        logger.info("Persistent cache cleared")

    def get_stats(self) -> dict:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "file_count": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }


def cached(cache_instance: Any, key_func: Optional[Callable] = None):
    """
    缓存装饰器

    Args:
        cache_instance: 缓存实例（LRUCache或PersistentCache）
        key_func: 生成缓存键的函数，默认使用参数的字符串表示

    Returns:
        装饰器函数

    Example:
        >>> cache = LRUCache(max_size=100)
        >>> @cached(cache)
        ... def expensive_function(x, y):
        ...     return x + y
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func is not None:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默认使用函数名和参数生成键
                args_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}:{args_str}"

            # 尝试从缓存获取
            cached_value = cache_instance.get(cache_key)
            if cached_value is not None:
                return cached_value

            # 计算结果并缓存
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result)

            return result

        return wrapper
    return decorator


# 全局缓存实例
_feature_cache = LRUCache(max_size=10000)
_prediction_cache = LRUCache(max_size=5000)
_structure_cache = PersistentCache(cache_dir="data/cache/structures")


def get_feature_cache() -> LRUCache:
    """获取特征缓存实例"""
    return _feature_cache


def get_prediction_cache() -> LRUCache:
    """获取预测缓存实例"""
    return _prediction_cache


def get_structure_cache() -> PersistentCache:
    """获取结构缓存实例"""
    return _structure_cache


def clear_all_caches():
    """清空所有缓存"""
    _feature_cache.clear()
    _prediction_cache.clear()
    _structure_cache.clear()
    logger.info("All caches cleared")


def get_all_cache_stats() -> dict:
    """
    获取所有缓存的统计信息

    Returns:
        统计信息字典
    """
    return {
        "feature_cache": _feature_cache.get_stats(),
        "prediction_cache": _prediction_cache.get_stats(),
        "structure_cache": _structure_cache.get_stats()
    }

