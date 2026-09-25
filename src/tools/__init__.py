"""Tools module for ASLK."""

# 延迟导入，避免循环导入问题
def __getattr__(name):
    if name == "BaseTool":
        from .base_tool import BaseTool
        return BaseTool
    elif name == "CrystaLLMWrapper":
        from .crystallm_wrapper import CrystaLLMWrapper
        return CrystaLLMWrapper
    elif name == "Ai4KappaWrapper":
        from .ai4kappa_wrapper import Ai4KappaWrapper
        return Ai4KappaWrapper
    elif name == "MattersimWrapper":
        from .mattersim_wrapper import MattersimWrapper
        return MattersimWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseTool",
    "CrystaLLMWrapper",
    "Ai4KappaWrapper",
    "MattersimWrapper",
]
