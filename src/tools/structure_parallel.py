"""
并行晶体结构生成模块
使用多进程并行处理多个组分的结构生成
"""

import logging
from typing import List, Dict, Any
import hashlib
import os
import shutil
import threading

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

# Limit BLAS/OMP threads before heavy imports in worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# 延迟导入，避免模块级别卡住
# 这些导入会在函数内部进行
_imports_done = False
_imports_lock = threading.Lock()
PMGComposition = None
CrystaLLMWrapper = None
Composition = None


def _derive_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)

def _ensure_imports():
    """Ensure the lazily imported worker dependencies are initialized once."""
    global _imports_done, PMGComposition, CrystaLLMWrapper, Composition
    if _imports_done:
        return

    with _imports_lock:
        if _imports_done:
            return

        from pymatgen.core import Composition as PMGComp
        PMGComposition = PMGComp

        try:
            from tools.crystallm_wrapper import CrystaLLMWrapper as Wrapper
        except ImportError:
            try:
                from crystallm_wrapper import CrystaLLMWrapper as Wrapper
            except ImportError:
                from src.tools.crystallm_wrapper import CrystaLLMWrapper as Wrapper
        CrystaLLMWrapper = Wrapper

        try:
            from utils.types import Composition as Comp
        except ImportError:
            try:
                from tools.types import Composition as Comp
            except ImportError:
                from src.utils.types import Composition as Comp
        Composition = Comp
        _imports_done = True


def _is_structural_generation_failure(result: Any) -> bool:
    """Return whether a failed result is local to the sampled structure."""
    metadata = getattr(result, "metadata", {}) or {}
    failure_kind = metadata.get("failure_kind")
    if failure_kind is not None:
        return failure_kind == "structure_invalid"
    error_text = str(getattr(result, "error", "") or result or "").lower()
    return any(
        marker in error_text
        for marker in (
            "returned no structures",
            "returned 0 valid structures",
            "valid structures; expected",
            "invalid cif",
            "no cif files were generated",
            "postprocessing produced no cif",
            "frontend structure conversion failed",
        )
    )


def _archive_generation_attempt(base_output_dir: str, formula: str, attempt: int) -> None:
    """Preserve a failed canonical attempt without deleting any artifacts."""
    source = os.path.join(base_output_dir, formula)
    if not os.path.isdir(source):
        return
    archive = os.path.join(base_output_dir, ".crystallm_attempts", formula, f"attempt_{attempt}")
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    shutil.copytree(source, archive, dirs_exist_ok=True)


def _promote_generation_attempt(attempt_output_dir: str, base_output_dir: str, formula: str) -> list[str]:
    """Copy a successful isolated attempt into the canonical formula directory."""
    source = os.path.join(attempt_output_dir, formula)
    target = os.path.join(base_output_dir, formula)
    if not os.path.isdir(source):
        raise FileNotFoundError(f"CrystaLLM attempt output is missing: {source}")
    if os.path.isdir(target):
        shutil.rmtree(target)
    for root, _, filenames in os.walk(source):
        relative_root = os.path.relpath(root, source)
        destination_root = target if relative_root == "." else os.path.join(target, relative_root)
        os.makedirs(destination_root, exist_ok=True)
        for filename in filenames:
            shutil.copy2(os.path.join(root, filename), os.path.join(destination_root, filename))
    processed_dir = os.path.join(target, "processed")
    return sorted(
        filename for filename in os.listdir(processed_dir)
        if filename.lower().endswith(".cif")
    ) if os.path.isdir(processed_dir) else []


def _validate_single_generated_cif(
    cif_path: str | os.PathLike,
    formula: str,
) -> tuple[bool, str]:
    """Check only that a generated CIF is parseable by downstream readers.

    The ``formula`` argument is retained for API and resume compatibility.  A
    parseable CIF is intentionally accepted here even when its composition or
    geometry is scientifically questionable; those outcomes belong to the
    relaxation/phonon result records rather than this pre-relaxation gate.
    """
    del formula
    try:
        from ase.io import read
        from pymatgen.io.cif import CifParser

        structures = CifParser(str(cif_path)).parse_structures(primitive=False)
        if not structures:
            return False, f"CIF parser returned no structure: {cif_path}"
        atoms = read(str(cif_path), format="cif")
        if len(atoms) == 0:
            return False, f"ASE CIF parser returned no atoms: {cif_path}"
    except Exception as exc:
        return False, f"CIF parsing failed: {type(exc).__name__}: {exc}"

    return True, ""


def _filter_valid_generated_cif_outputs(
    cif_paths: list[str | os.PathLike],
    formula: str,
) -> tuple[list[str | os.PathLike], list[str]]:
    """Keep parseable CIFs so one malformed sample does not discard the whole material."""
    valid_paths: list[str | os.PathLike] = []
    errors: list[str] = []
    for cif_path in cif_paths:
        is_valid, error = _validate_single_generated_cif(cif_path, formula)
        if is_valid:
            valid_paths.append(cif_path)
        else:
            errors.append(error)
    return valid_paths, errors


def _validate_generated_cif_outputs(
    cif_paths: list[str | os.PathLike],
    formula: str,
    expected_count: int,
) -> tuple[bool, str]:
    """Validate the exact CIF set used by strict artifact recovery checks."""
    if len(cif_paths) != expected_count:
        return False, f"CIF count {len(cif_paths)} does not match expected {expected_count}"

    for cif_path in cif_paths:
        is_valid, error = _validate_single_generated_cif(cif_path, formula)
        if not is_valid:
            return False, error
    return True, ""


def generate_single_composition_worker(args: tuple) -> Dict[str, Any]:
    """
    为单个组分生成结构
    
    Args:
        args: (index, material, wrapper_config, gen_config)
    
    Returns:
        包含生成结果的字典
    """
    # 在worker中确保导入完成
    _ensure_imports()
    
    i, material, wrapper_config, gen_config = args
    
    try:
        # 使用模块级 logger，避免在主进程中重复调用 basicConfig
        formula = material.get('formula', '')
        if not formula:
            return {
                'index': i,
                'formula': 'Unknown',
                'success': False,
                'status': 'failed',
                'fatal': True,
                'error': 'No formula provided'
            }
        
        logger.info(f"[Task {i+1}] 开始生成 {formula} 的结构...")
        
        # 解析元素组成（使用模块级别的PMGComposition）
        pmg_comp = PMGComposition(formula)
        elements = {str(el): amt for el, amt in pmg_comp.get_el_amt_dict().items()}
        composition = Composition(formula=formula, elements=elements)
        logger.info(f"[Task {i+1}] 元素组成解析完成: {elements}")
        
        base_output_dir = str(wrapper_config.get("output_dir") or "")
        requested_structures = int(gen_config.get("n_structures", 0))
        task_seed = gen_config.get("seed")
        attempt_records = []

        def _collect_candidate_cifs(attempt_output_dir: str) -> list[str]:
            processed_dir = os.path.join(attempt_output_dir, formula, "processed")
            if not os.path.isdir(processed_dir):
                return []
            return sorted(
                os.path.join(processed_dir, filename)
                for filename in os.listdir(processed_dir)
                if filename.lower().endswith(".cif")
            )

        def _accept_valid_cifs(
            attempt_output_dir: str,
            attempt: int,
            attempt_seed: int,
            backend: str,
            n_relaxed: int,
            reported_count: int,
            candidate_cifs: list[str],
            valid_cifs: list[str | os.PathLike],
            validation_errors: list[str] | None = None,
        ) -> Dict[str, Any]:
            valid_paths = {os.path.abspath(os.fspath(path)) for path in valid_cifs}
            for candidate in candidate_cifs:
                if os.path.abspath(candidate) not in valid_paths:
                    try:
                        os.remove(candidate)
                    except OSError:
                        pass
            if attempt > 1:
                promoted_cifs = _promote_generation_attempt(
                    attempt_output_dir,
                    base_output_dir,
                    formula,
                )
            else:
                promoted_cifs = [os.path.basename(path) for path in valid_cifs]
            accepted_count = len(promoted_cifs)
            rejected_cif_count = max(0, len(candidate_cifs) - accepted_count)
            precheck_status = (
                "passed"
                if accepted_count > 0 and rejected_cif_count == 0
                else "partial"
                if accepted_count > 0
                else "failed"
            )
            if accepted_count < requested_structures:
                logger.warning(
                    f"[{i+1}] {formula}: 接受部分有效结构 "
                    f"({accepted_count}/{requested_structures})，无效 CIF 已隔离"
                )
            logger.info(
                f"[{i+1}] {formula}: 成功生成 {accepted_count} 个有效结构 "
                f"(CrystaLLM reported={reported_count}, attempt={attempt})"
            )
            result_status = (
                "success_partial"
                if accepted_count < requested_structures
                else "success"
            )
            return {
                'index': i,
                'formula': formula,
                'success': True,
                'status': result_status,
                'n_structures': accepted_count,
                'requested_structures': requested_structures,
                'partial': accepted_count < requested_structures,
                'n_relaxed': n_relaxed,
                'generator_backend': backend,
                'generation_attempt': attempt,
                'seed': attempt_seed,
                'cif_files': promoted_cifs,
                'precheck_status': precheck_status,
                'parseable_cif_count': accepted_count,
                'parse_failed_cif_count': rejected_cif_count,
                'rejected_cif_count': rejected_cif_count,
                'validation_errors': list(validation_errors or []),
                'attempts': attempt_records + [{
                    "attempt": attempt,
                    "seed": attempt_seed,
                    "status": "success_partial" if accepted_count < requested_structures else "success",
                }],
            }

        for attempt in (1, 2):
            attempt_seed = (
                task_seed
                if attempt == 1
                else _derive_seed(task_seed, "crystallm_seed_retry", formula, requested_structures)
            )
            attempt_output_dir = base_output_dir
            if attempt > 1:
                _archive_generation_attempt(base_output_dir, formula, 1)
                attempt_output_dir = os.path.join(
                    base_output_dir,
                    ".crystallm_attempts",
                    formula,
                    f"attempt_{attempt}",
                )
            attempt_wrapper_config = dict(wrapper_config)
            attempt_wrapper_config["output_dir"] = attempt_output_dir

            try:
                wrapper = CrystaLLMWrapper(**attempt_wrapper_config)
                logger.info(f"[Task {i+1}] Wrapper 初始化成功 (attempt={attempt}, seed={attempt_seed})")
            except Exception as init_error:
                logger.error(f"[Task {i+1}] Wrapper 初始化失败: {init_error}")
                raise

            attempt_config = dict(gen_config)
            attempt_config["seed"] = attempt_seed
            try:
                result = wrapper.run(composition, **attempt_config)
                logger.info(f"[Task {i+1}] wrapper.run() 完成 (attempt={attempt})")
            except Exception as run_error:
                logger.error(f"[Task {i+1}] wrapper.run() 失败 (attempt={attempt}): {run_error}")
                raise

            if result.is_success():
                reported_count = len(result.result or [])
                n_relaxed = result.metadata.get('n_relaxed', 0)
                backend = result.metadata.get("generator_backend")
                if backend != "CrystaLLM":
                    return {
                        'index': i,
                        'formula': formula,
                        'success': False,
                        'status': 'failed',
                        'fatal': True,
                        'error': f"Unexpected structure generator backend: {backend!r}",
                    }

                candidate_cifs = _collect_candidate_cifs(attempt_output_dir)
                strict_valid, strict_error = _validate_generated_cif_outputs(
                    candidate_cifs,
                    formula,
                    requested_structures,
                )
                if strict_valid:
                    valid_cifs = candidate_cifs
                    validation_errors = []
                else:
                    valid_cifs, validation_errors = _filter_valid_generated_cif_outputs(
                        candidate_cifs,
                        formula,
                    )
                if valid_cifs:
                    return _accept_valid_cifs(
                        attempt_output_dir,
                        attempt,
                        attempt_seed,
                        backend,
                        n_relaxed,
                        reported_count,
                        candidate_cifs,
                        valid_cifs,
                        validation_errors,
                    )

                result_error = (
                    validation_errors[0]
                    if validation_errors
                    else strict_error
                    or f"CrystaLLM generated {reported_count} structures; "
                    f"expected at least 1 valid structure"
                )
                attempt_records.append({
                    "attempt": attempt,
                    "seed": attempt_seed,
                    "status": "structure_invalid",
                    "error": result_error,
                })
                logger.warning(
                    f"[{i+1}] {formula}: 最终 CIF 校验失败 "
                    f"(attempt={attempt})，不会进入下游: {result_error}"
                )
                if attempt == 1:
                    continue
                return {
                    'index': i,
                    'formula': formula,
                    'success': False,
                    'status': 'skipped',
                    'skipped': True,
                    'fatal': False,
                    'skip_reason': 'structure_invalid_after_seed_retry',
                    'error': result_error,
                    'attempts': attempt_records,
                }

            error = result.error or "Unknown CrystaLLM generation error"
            if _is_structural_generation_failure(result):
                candidate_cifs = _collect_candidate_cifs(attempt_output_dir)
                valid_cifs, validation_errors = _filter_valid_generated_cif_outputs(
                    candidate_cifs,
                    formula,
                )
                if valid_cifs:
                    backend = result.metadata.get("generator_backend") or "CrystaLLM"
                    return _accept_valid_cifs(
                        attempt_output_dir,
                        attempt,
                        attempt_seed,
                        backend,
                        result.metadata.get("n_relaxed", 0),
                        len(result.result or []),
                        candidate_cifs,
                        valid_cifs,
                        validation_errors,
                    )
            attempt_records.append({
                "attempt": attempt,
                "seed": attempt_seed,
                "status": result.metadata.get("failure_kind", "failed"),
                "error": str(error),
            })
            if _is_structural_generation_failure(result) and attempt == 1:
                logger.warning(
                    f"[{i+1}] {formula}: 结构局部失败，使用新 seed 重试: {error}"
                )
                continue
            if _is_structural_generation_failure(result) and attempt == 2:
                logger.warning(f"[{i+1}] {formula}: 两次结构生成均失败，跳过当前材料")
                return {
                    'index': i,
                    'formula': formula,
                    'success': False,
                    'status': 'skipped',
                    'skipped': True,
                    'fatal': False,
                    'skip_reason': 'structure_invalid_after_seed_retry',
                    'error': str(error),
                    'attempts': attempt_records,
                }
            return {
                'index': i,
                'formula': formula,
                'success': False,
                'status': 'failed',
                'fatal': True,
                'error': str(error),
                'attempts': attempt_records,
            }

        raise RuntimeError(f"CrystaLLM generation attempts exhausted for {formula}")
            
    except Exception as e:
        import traceback
        err_msg = f"组分 {i+1} ({material.get('formula', 'Unknown')}) 生成异常: {e}"
        logger.error(err_msg)
        logger.debug(traceback.format_exc())
        return {
            'index': i,
            'formula': material.get('formula', 'Unknown'),
            'success': False,
            'status': 'failed',
            'fatal': True,
            'error': str(e)
        }


def generate_structures_parallel(
    materials: List[Dict],
    device: str = "cuda",
    output_dir: str = None,
    n_structures: int = 5,
    relax_structures: bool = True,
    pressure: float = 0.0,
    relax_output_dir: str = None,
    max_workers: int = 4,
    gpus: List[str] = None,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """
    并行生成多个组分的结构
    
    Args:
        materials: 材料列表
        device: 默认设备（向后兼容）
        output_dir: 输出目录
        n_structures: 每个材料生成的结构数
        relax_structures: 是否弛豫
        pressure: 弛豫压力
        relax_output_dir: 弛豫输出目录
        max_workers: 最大并行数
        gpus: GPU列表，如果为None则使用device参数
        seed: 随机种子基准值
    
    Returns:
        结果列表
    """
    try:
        from utils.gpu_parallel import (
            normalize_gpu_list,
            run_tasks_by_gpu,
            validate_gpu_devices,
        )
    except ImportError:
        from src.utils.gpu_parallel import (
            normalize_gpu_list,
            run_tasks_by_gpu,
            validate_gpu_devices,
        )

    gpus = normalize_gpu_list(gpus, device=device)
    validate_gpu_devices(gpus)
    logger.info(f"📊 GPU配置: {gpus} ({len(gpus)}个)")
    logger.info(f"准备生成 {len(materials)} 个材料的结构 (max_workers={max_workers})")

    tasks = []
    for i, m in enumerate(materials):
        assigned_gpu = gpus[i % len(gpus)]
        wrapper_config = {'device': assigned_gpu, 'output_dir': output_dir}
        task_seed = _derive_seed(seed, "structure_generation", m.get('formula', ''), i, n_structures)
        gen_config = {
            'n_structures': n_structures,
            'relax_structures': relax_structures,
            'pressure': pressure,
            'relax_output_dir': relax_output_dir,
            'calculate_properties': False,
            'seed': task_seed,
        }
        tasks.append((i, m, wrapper_config, gen_config))
        logger.info(
            f"  任务 {i+1}: {m.get('formula', 'Unknown')} -> "
            f"{assigned_gpu} (seed={task_seed})"
        )

    def _run_generation(task):
        try:
            return generate_single_composition_worker(task)
        except Exception as exc:
            index = task[0]
            return {
                'index': index,
                'formula': task[1].get('formula', 'Unknown'),
                'success': False,
                'status': 'failed',
                'fatal': True,
                'error': str(exc),
            }

    # A single visible GPU remains serialized for safety. With multiple GPUs,
    # each GPU gets its own bounded lane and each task launches its own model
    # subprocess with an explicit device.
    workers_per_gpu = max(1, int(max_workers)) if len(gpus) > 1 else 1
    actual_workers = len(gpus) * workers_per_gpu
    if len(gpus) > 1:
        logger.info("🚀 使用多GPU并行模式")
    else:
        logger.info("使用单GPU顺序模式")
    logger.info(f"  - GPU数量: {len(gpus)}")
    logger.info(f"  - 每GPU并行数: {workers_per_gpu}")
    logger.info(f"  - 总并行数: {actual_workers}")

    results = run_tasks_by_gpu(
        tasks,
        gpus=gpus,
        workers_per_gpu=workers_per_gpu,
        task_device=lambda task: task[2]['device'],
        run_task=_run_generation,
    )
    for idx, result in enumerate(results):
        formula = tasks[idx][1].get('formula', 'Unknown')
        assigned_gpu = tasks[idx][2]['device']
        if result.get('success'):
            print(f"[{idx+1}/{len(tasks)}] ✅ {formula} ({assigned_gpu}) 生成成功")
        else:
            print(
                f"[{idx+1}/{len(tasks)}] ❌ {formula} ({assigned_gpu}) "
                f"生成失败: {result.get('error')}"
            )

    return results
