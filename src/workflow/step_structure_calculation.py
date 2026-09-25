# -*- coding: utf-8 -*-
"""
Workflow Step 4: Structure Generation and Calculation
负责结构生成、弛豫、声子计算（记录到 CSV）、结构去重（避免重复计算）、
以及热导率计算。声子计算与弛豫阶段合并执行，不再单独补算。
"""
import os
import sys
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

# 所有 pymatgen 相关导入都延迟到函数内部
# 这是为了避免 Windows 上 torch/pymatgen 导入顺序冲突

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from tools.structure_parallel import generate_structures_parallel
from utils.gpu_parallel import (
    SpawnTaskError,
    SpawnTaskStartError,
    normalize_gpu_list,
    run_spawn_task,
    run_tasks_by_gpu,
    set_process_cuda_device,
    validate_gpu_devices,
)
from utils.types import Composition, CrystalStructure


def safe_clear_memory(device="cuda"):
    """
    安全清理内存，并捕获可能的 OOM 错误。

    Args:
        device: 设备类型
    """
    import gc
    gc.collect()

    if str(device).startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("  [WARN] GPU 清理时发生 OOM，尝试强制重置...")
                        try:
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.reset_accumulated_memory_stats()
                            torch.cuda.empty_cache()
                        except Exception:
                            print("  [WARN] Forced GPU reset also failed; GPU may need restart")
                    else:
                        raise
        except ImportError:
            pass
        except Exception as e:
            print(f"  [WARN] 内存清理告警: {e}")


def normalize_formula(formula: str) -> str:
    """Convert Unicode subscripts in a formula to normal digits."""
    subscript_map = {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
    normalized = formula
    for subscript, normal in subscript_map.items():
        normalized = normalized.replace(subscript, normal)
    return normalized


def _derive_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


def _detect_space_group(structure) -> str:
    """Detect the space-group symbol from the actual structure geometry."""
    try:
        return structure.get_space_group_info(symprec=0.01, angle_tolerance=5.0)[0]
    except Exception:
        return "Unknown"


def _apply_task_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _is_cuda_oom_text(*values: object) -> bool:
    text = " ".join(str(value or "") for value in values).lower()
    return "out of memory" in text or "cuda oom" in text


def _is_valid_relaxed_cif(path: Path, formula: str | None = None) -> tuple[bool, str]:
    """Validate only that an ASE-serialized CIF can be parsed downstream.

    ``formula`` remains in the signature for resume/test compatibility, but a
    relaxed structure is not required to preserve the requested composition.
    The requested composition is the material-directory key; the actual
    structure formula is derived from the parsed CIF by the caller.
    """
    del formula
    try:
        from pymatgen.io.cif import CifParser

        structures = CifParser(str(path)).parse_structures(primitive=False)
        if not structures:
            return False, "Invalid CIF file with no structures"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _actual_formula_from_relaxed_cif(path: Path) -> str:
    """Read a relaxed CIF's actual formula for persisted lineage metadata."""
    try:
        from pymatgen.core import Structure

        return Structure.from_file(str(path)).composition.reduced_formula
    except Exception:
        return ""


def _is_valid_thermal_csv(csv_file: Path, expected_cifs: set[str] | None = None) -> bool:
    required_column = "Kappa_Slack (W m-1 K-1)"
    if not csv_file.exists():
        return False
    try:
        with open(csv_file, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if required_column not in fieldnames:
                return False
            rows = list(reader)
            if not rows or not any(str(row.get(required_column) or "").strip() for row in rows):
                return False
            if expected_cifs:
                if "CIF_File" not in fieldnames:
                    return False
                written_cifs = {str(row.get("CIF_File") or "").strip() for row in rows}
                return expected_cifs.issubset(written_cifs)
            return True
    except (OSError, csv.Error, UnicodeError):
        return False


def relax_structure_worker(args):
    """Relax one structure and run phonon calculation."""
    cif_path, formula, relax_base_dir, pressure, gpu_device, task_seed = args
    import os

    if os.name == "nt":
        if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").lower():
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    worker_device = str(gpu_device or "cuda").strip().lower()
    if worker_device.startswith("cuda"):
        # Keep the logical CUDA index unchanged. The child sees the same
        # visible-device namespace as the parent and all model constructors
        # receive the explicit assigned device (cuda:N).
        set_process_cuda_device(worker_device)
        print(f"  [Worker {os.getpid()}] Using GPU: {worker_device}")
    _apply_task_seed(task_seed)

    try:
        import torch
        import gc
        import time

        if torch.cuda.is_available():
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            time.sleep(1)
    except Exception as e:
        print(f"  [Worker] GPU cleanup warning: {e}")

    mattersim = None
    try:
        from tools.mattersim_wrapper import MattersimWrapper
        from ase.io import write as ase_write
        from io import StringIO
        from pathlib import Path
        from pymatgen.core import Composition as PMGComposition, Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.io.vasp import Poscar as PmgPoscar

        cif_path = Path(cif_path)
        relax_base_dir = Path(relax_base_dir)

        comp_dir = relax_base_dir / formula
        comp_dir.mkdir(parents=True, exist_ok=True)

        output_file = comp_dir / cif_path.name

        pmg_struct = Structure.from_file(str(cif_path))
        pmg_comp = PMGComposition(formula)
        elements = {str(el): amt for el, amt in pmg_comp.get_el_amt_dict().items()}
        comp_obj = Composition(formula=formula, elements=elements)

        poscar_str = PmgPoscar(pmg_struct).get_string()
        l = pmg_struct.lattice
        lattice_dict = {
            "a": l.a, "b": l.b, "c": l.c,
            "alpha": l.alpha, "beta": l.beta, "gamma": l.gamma,
        }

        crystal_struct = CrystalStructure(
            structure_id=cif_path.stem,
            composition=comp_obj,
            poscar=poscar_str,
            lattice_params=lattice_dict,
            n_atoms=len(pmg_struct),
            space_group=_detect_space_group(pmg_struct),
        )

        mattersim = MattersimWrapper(config={"device": worker_device or "cuda"})
        print(f"  Relaxing: {cif_path.name}")
        response = mattersim.relax_structure(crystal_struct, pressure=pressure)

        if response.is_success():
            relaxed_atoms = response.result
            relaxed_pmg_struct = AseAtomsAdaptor.get_structure(relaxed_atoms)
            relaxed_formula = relaxed_pmg_struct.composition.reduced_formula
            requested_formula = PMGComposition(formula).reduced_formula
            composition_mismatch = relaxed_formula != requested_formula
            if composition_mismatch:
                print(
                    "  [WARN] Relaxed composition differs from requested formula: "
                    f"requested={requested_formula}, actual={relaxed_formula}"
                )
            ase_write(str(output_file), relaxed_atoms, format="cif")
            serialized_valid, serialized_error = _is_valid_relaxed_cif(output_file)
            if not serialized_valid:
                print(f"  Relaxation output invalid: {output_file} - {serialized_error}")
                return {
                    "success": False,
                    "formula": relaxed_formula,
                    "composition": formula,
                    "relaxed_formula": relaxed_formula,
                    "composition_mismatch": composition_mismatch,
                    "cif_file": cif_path.name,
                    "file": str(output_file),
                    "error": f"Invalid relaxed CIF: {serialized_error}",
                    "relax_error": f"Invalid relaxed CIF: {serialized_error}",
                    "phonon_success": False,
                    "phonon_error": "Skipped because relaxed CIF validation failed",
                }
            print(f"  Relaxation succeeded: {output_file}")

            phonon_success = False
            phonon_error = None
            has_imaginary = "Unknown"
            min_frequency = None
            gamma_min_optical = None
            gamma_max_acoustic = None

            try:
                poscar_io = StringIO()
                ase_write(poscar_io, relaxed_atoms, format="vasp")
                poscar_relaxed = poscar_io.getvalue()
                cell = relaxed_atoms.get_cell()
                lengths = cell.lengths()
                angles = cell.angles()
                relaxed_struct = CrystalStructure(
                    structure_id=cif_path.stem,
                    composition=comp_obj,
                    poscar=poscar_relaxed,
                    lattice_params={
                        "a": float(lengths[0]),
                        "b": float(lengths[1]),
                        "c": float(lengths[2]),
                        "alpha": float(angles[0]),
                        "beta": float(angles[1]),
                        "gamma": float(angles[2]),
                    },
                    n_atoms=len(relaxed_atoms),
                    space_group=_detect_space_group(relaxed_pmg_struct),
                )

                phonon_dir = comp_dir / f"{cif_path.stem}_phonon"
                phonon_dir.mkdir(parents=True, exist_ok=True)
                plot_path = phonon_dir / "phonon_spectrum.png"

                phonon_resp = mattersim.run(
                    relaxed_struct,
                    calculate_phonon=True,
                    save_plot=True,
                    plot_path=str(plot_path),
                )
                if phonon_resp.is_success():
                    phonon_success = True
                    ph_result = phonon_resp.result
                    if ph_result is not None and ph_result.has_imaginary_freq is not None:
                        has_imaginary = "Y" if ph_result.has_imaginary_freq else "N"
                    min_frequency = getattr(ph_result, "min_frequency", None)
                    gamma_min_optical = getattr(ph_result, "gamma_min_optical", None)
                    gamma_max_acoustic = getattr(ph_result, "gamma_max_acoustic", None)
                else:
                    phonon_error = phonon_resp.error
            except Exception as phonon_exc:
                if _is_cuda_oom_text(phonon_exc):
                    raise
                phonon_error = str(phonon_exc)

            return {
                "success": True,
                "formula": relaxed_formula,
                "composition": formula,
                "relaxed_formula": relaxed_formula,
                "composition_mismatch": composition_mismatch,
                "file": str(output_file),
                "cif_file": cif_path.name,
                "relax_error": None,
                "phonon_success": phonon_success,
                "phonon_error": phonon_error,
                "has_imaginary": has_imaginary,
                "min_frequency": min_frequency,
                "gamma_min_optical": gamma_min_optical,
                "gamma_max_acoustic": gamma_max_acoustic,
            }

        error = response.error or "MatterSim relaxation failed"
        print(f"  Relaxation failed: {cif_path.name} - {error}")
        return {
            "success": False,
            "formula": None,
            "composition": formula,
            "cif_file": cif_path.name,
            "error": "CUDA OOM" if _is_cuda_oom_text(error) else error,
            "skipped": _is_cuda_oom_text(error),
        }

    except Exception as e:
        import traceback

        error_msg = "CUDA OOM" if _is_cuda_oom_text(e) else str(e)
        error_traceback = traceback.format_exc()
        if _is_cuda_oom_text(e):
            print(f"  WARNING: Relaxation OOM: {cif_path.name} - insufficient GPU memory; skipping structure")
            return {
                "success": False,
                "formula": None,
                "composition": formula,
                "cif_file": Path(cif_path).name,
                "file": str(output_file) if "output_file" in locals() and output_file.exists() else None,
                "error": error_msg,
                "error_type": type(e).__name__,
                "traceback": error_traceback,
                "skipped": True,
            }
        print(f"  Relaxation error: {e}")
        return {
            "success": False,
            "formula": None,
            "composition": formula,
            "cif_file": Path(cif_path).name,
            "file": str(output_file) if "output_file" in locals() and output_file.exists() else None,
            "error": error_msg,
            "error_type": type(e).__name__,
            "traceback": error_traceback,
        }
    finally:
        try:
            if mattersim is not None:
                mattersim.close()
        except Exception:
            pass
        try:
            import torch
            import gc
            import time
            if torch.cuda.is_available():
                for _ in range(3):
                    gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                time.sleep(0.5)
                gc.collect()
        except Exception:
            pass


def run_relax_task_with_timeout(
    task,
    timeout_sec: int,
    prefer_subprocess: bool | None = None,
    allow_in_process_fallback: bool = True,
):
    """Run one relaxation+phonon task with an isolated CUDA context."""
    from pathlib import Path as _Path

    worker_device = str(task[4] or "cuda").strip().lower()

    def _failure(
        error: object,
        *,
        error_type: str | None = None,
        diagnostic_path: _Path | None = None,
        exit_code: int | None = None,
        traceback_text: str | None = None,
    ) -> dict:
        result = {
            "success": False,
            "formula": None,
            "composition": task[1],
            "cif_file": _Path(task[0]).name,
            "error": str(error),
        }
        if error_type:
            result["worker_error_type"] = error_type
        if diagnostic_path:
            result["worker_diagnostic"] = str(diagnostic_path)
        if exit_code is not None:
            result["worker_exit_code"] = exit_code
        if traceback_text:
            result["traceback"] = traceback_text
        elif getattr(error, "traceback_text", None):
            result["traceback"] = error.traceback_text
        if diagnostic_path and result.get("traceback"):
            try:
                diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                with diagnostic_path.open("a", encoding="utf-8", newline="") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "event": "relax_task_failure",
                                "error_type": result.get("worker_error_type"),
                                "error": result["error"],
                                "exit_code": result.get("worker_exit_code"),
                                "traceback": result["traceback"],
                            },
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
            except OSError:
                pass
        return result

    diagnostic_path = (
        _Path(task[2])
        / str(task[1])
        / "worker_diagnostics"
        / f"{_Path(task[0]).stem}.worker.log"
    )

    def _run_in_process(diagnostic: _Path | None = None):
        try:
            return relax_structure_worker(task)
        except BaseException as exc:
            import traceback

            return _failure(
                exc,
                error_type=type(exc).__name__,
                diagnostic_path=diagnostic,
                traceback_text=traceback.format_exc(),
            )

    if worker_device.startswith("cuda"):
        try:
            validate_gpu_devices([worker_device])
        except BaseException as exc:
            import traceback

            return _failure(
                exc,
                error_type=type(exc).__name__,
                traceback_text=traceback.format_exc(),
            )

    if prefer_subprocess is None:
        prefer_subprocess = os.name != "nt"
    if not prefer_subprocess:
        return _run_in_process(diagnostic_path)

    try:
        return run_spawn_task(
            relax_structure_worker,
            task,
            timeout_sec=timeout_sec,
            diagnostic_path=diagnostic_path,
        )
    except TimeoutError:
        result = _failure(f"Timeout; worker diagnostic: {diagnostic_path}")
        result["worker_diagnostic"] = str(diagnostic_path)
        result["worker_error_type"] = "TimeoutError"
        return result
    except SpawnTaskStartError as exc:
        if allow_in_process_fallback:
            print(
                "  [WARN] Failed to start isolated relax worker, "
                f"falling back in-process: {exc}"
            )
            return _run_in_process(diagnostic_path)
        return _failure(
            f"Failed to start worker process: {exc}",
            error_type="SpawnTaskStartError",
            diagnostic_path=diagnostic_path,
        )
    except SpawnTaskError as exc:
        return _failure(
            exc,
            error_type="SpawnTaskError",
            diagnostic_path=diagnostic_path,
            exit_code=getattr(exc, "exit_code", None),
        )


def calculate_kappa_worker(args):
    """Calculate thermal conductivity for eligible structures on one device."""
    if len(args) == 3:
        comp_dir, formula, task_seed = args
        worker_device = "cuda"
        expected_cifs = set()
    elif len(args) == 5:
        comp_dir, formula, task_seed, worker_device, expected_cifs = args
    else:
        raise ValueError(f"Unexpected thermal task shape: {args!r}")

    worker_device = str(worker_device or "cuda").strip().lower()
    if worker_device.startswith("cuda"):
        set_process_cuda_device(worker_device)

    import time
    time.sleep(0.5)
    _apply_task_seed(task_seed)
    wrapper = None

    try:
        from pathlib import Path
        from pymatgen.core import Composition as PMGComposition, Structure
        from pymatgen.io.vasp import Poscar as PmgPoscar

        comp_dir = Path(comp_dir)
        cif_files = sorted(
            path for path in comp_dir.glob("*.cif")
            if not expected_cifs or path.name in expected_cifs
        )
        if not cif_files:
            return {"success": False, "formula": formula, "error": "No eligible CIF files found"}

        pmg_comp = PMGComposition(formula)
        elements = {str(el): amt for el, amt in pmg_comp.get_el_amt_dict().items()}
        composition = Composition(formula=formula, elements=elements)

        structures = []
        for cif_file in cif_files:
            try:
                pmg_s = Structure.from_file(str(cif_file))
                poscar_str = PmgPoscar(pmg_s).get_string()

                l = pmg_s.lattice
                lattice_dict = {
                    "a": l.a, "b": l.b, "c": l.c,
                    "alpha": l.alpha, "beta": l.beta, "gamma": l.gamma,
                }

                cs = CrystalStructure(
                    structure_id=cif_file.stem,
                    composition=composition,
                    poscar=poscar_str,
                    lattice_params=lattice_dict,
                    n_atoms=len(pmg_s),
                    space_group=_detect_space_group(pmg_s),
                )
                structures.append(cs)
            except Exception as e:
                print(f"  读取 CIF 失败 {cif_file}: {e}")

        if not structures:
            return {"success": False, "formula": formula, "error": "No valid structures loaded"}

        from tools.crystallm_wrapper import CrystaLLMWrapper as _CrystaLLMWrapper
        wrapper = _CrystaLLMWrapper(output_dir=str(comp_dir), device=worker_device)
        print(f"  正在计算热导率 {formula}（{len(structures)} 个结构，设备={worker_device}）...")
        wrapper._calculate_and_save_thermal_conductivity(comp_dir, structures, composition)

        thermal_csv = comp_dir / "thermal_conductivity.csv"
        expected_cifs = {path.name for path in cif_files}
        if not _is_valid_thermal_csv(thermal_csv, expected_cifs):
            return {
                "success": False,
                "formula": formula,
                "error": "Thermal conductivity CSV is missing or invalid",
            }
        return {"success": True, "formula": formula, "thermal_csv": str(thermal_csv)}

    except BaseException as e:
        import traceback

        error = "CUDA OOM" if _is_cuda_oom_text(e) else str(e)
        error_traceback = traceback.format_exc()
        print(f"  热导率计算异常 {formula} - {error}")
        return {
            "success": False,
            "formula": formula,
            "error": error,
            "worker_error_type": type(e).__name__,
            "traceback": error_traceback,
            "skipped": _is_cuda_oom_text(e),
        }
    finally:
        if wrapper is not None:
            try:
                wrapper.close()
            except Exception:
                pass
        safe_clear_memory(worker_device)


def run_kappa_task_with_timeout(
    task,
    timeout_sec: float | None = None,
):
    """Run one thermal-conductivity material in an isolated spawn process."""
    from pathlib import Path as _Path

    worker_device = str(task[3] or "cuda").strip().lower()
    diagnostic_path = (
        _Path(task[0]) / "worker_diagnostics" / "thermal.worker.log"
    )
    try:
        if worker_device.startswith("cuda"):
            validate_gpu_devices([worker_device])
        result = run_spawn_task(
            calculate_kappa_worker,
            task,
            timeout_sec=timeout_sec,
            diagnostic_path=diagnostic_path,
        )
        if isinstance(result, dict) and not result.get("success"):
            result.setdefault("worker_diagnostic", str(diagnostic_path))
        return result
    except TimeoutError:
        return {
            "success": False,
            "formula": task[1],
            "error": f"Timeout after {timeout_sec}s",
            "worker_error_type": "TimeoutError",
            "worker_diagnostic": str(diagnostic_path),
        }
    except SpawnTaskStartError as exc:
        return {
            "success": False,
            "formula": task[1],
            "error": str(exc),
            "worker_error_type": "SpawnTaskStartError",
            "cif_file": _Path(task[0]).name,
            "worker_diagnostic": str(diagnostic_path),
        }
    except SpawnTaskError as exc:
        return {
            "success": False,
            "formula": task[1],
            "error": str(exc),
            "worker_error_type": "SpawnTaskError",
            "worker_exit_code": getattr(exc, "exit_code", None),
            "traceback": getattr(exc, "traceback_text", None),
            "cif_file": _Path(task[0]).name,
            "worker_diagnostic": str(
                getattr(exc, "diagnostic_path", None) or diagnostic_path
            ),
        }


def _load_latest_relax_rows(comp_dir: Path) -> dict[str, dict]:
    """Load the latest persisted result for each source CIF, keeping attempts auditable."""
    log_file = comp_dir / "relax_phonon_results.csv"
    latest: dict[str, dict] = {}
    if not log_file.exists():
        return latest
    try:
        with open(log_file, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                cif_name = str(row.get("CIF_File") or "").strip()
                if cif_name:
                    latest[cif_name] = row
    except Exception:
        return {}
    return latest


def _relaxed_cif_candidates(comp_dir: Path, row: dict) -> list[Path]:
    relaxed_cif = str(row.get("Relaxed_CIF") or "").strip()
    candidates = [Path(relaxed_cif)] if relaxed_cif else []
    if candidates and not candidates[0].is_absolute():
        candidates[0] = comp_dir / candidates[0]
    cif_name = str(row.get("CIF_File") or "").strip()
    if cif_name:
        candidates.append(comp_dir / cif_name)
    return candidates


def _load_relax_status(comp_dir: Path) -> tuple[set[str], set[str]]:
    """Return attempted and latest relaxation-successful CIF names from the log."""
    latest = _load_latest_relax_rows(comp_dir)
    attempted = set(latest)
    successful = {
        cif_name
        for cif_name, row in latest.items()
        if str(row.get("Relax_Success") or "").strip().upper() == "Y"
        or str(row.get("Relaxed_CIF") or "").strip()
    }
    return attempted, successful


def _load_completed_phonon_cifs(comp_dir: Path) -> set[str]:
    """Return source CIFs with persisted successful relaxation + phonon results."""
    log_file = comp_dir / "relax_phonon_results.csv"
    completed: set[str] = set()
    if not log_file.exists():
        return completed

    for cif_name, row in _load_latest_relax_rows(comp_dir).items():
        relax_ok = str(row.get("Relax_Success") or "").strip().upper() == "Y"
        phonon_ok = str(row.get("Phonon_Success") or "").strip().upper() == "Y"
        if not (relax_ok and phonon_ok):
            continue
        candidate = next(
            (path for path in _relaxed_cif_candidates(comp_dir, row) if path.exists()),
            None,
        )
        if candidate is not None and _is_valid_relaxed_cif(candidate, comp_dir.name)[0]:
            completed.add(cif_name)

    return completed


def _load_completed_phonon_relaxed_cifs(comp_dir: Path) -> set[str]:
    """Return persisted relaxed CIF filenames eligible for thermal calculation."""
    log_file = comp_dir / "relax_phonon_results.csv"
    completed: set[str] = set()
    if not log_file.exists():
        return completed

    for _, row in _load_latest_relax_rows(comp_dir).items():
        relax_ok = str(row.get("Relax_Success") or "").strip().upper() == "Y"
        phonon_ok = str(row.get("Phonon_Success") or "").strip().upper() == "Y"
        if not (relax_ok and phonon_ok):
            continue
        candidate = next(
            (path for path in _relaxed_cif_candidates(comp_dir, row) if path.exists()),
            None,
        )
        if candidate is not None and _is_valid_relaxed_cif(candidate, comp_dir.name)[0]:
            completed.add(candidate.name)

    return completed


def _load_oom_skipped_cifs(comp_dir: Path) -> set[str]:
    """Return CIFs deliberately skipped after a CUDA OOM."""
    log_file = comp_dir / "relax_phonon_results.csv"
    skipped: set[str] = set()
    if not log_file.exists():
        return skipped

    try:
        with open(log_file, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cif_name = str(row.get("CIF_File") or "").strip()
                error_text = " ".join(
                    str(row.get(key) or "")
                    for key in ("Relax_Error", "Phonon_Error")
                ).lower()
                if cif_name and ("out of memory" in error_text or "cuda oom" in error_text):
                    skipped.add(cif_name)
    except Exception:
        return set()

    return skipped


def _load_terminal_skipped_cifs(comp_dir: Path) -> set[str]:
    """Return CIFs with a persisted non-success result that must not retry forever."""
    log_file = comp_dir / "relax_phonon_results.csv"
    skipped: set[str] = set()
    if not log_file.exists():
        return skipped

    for cif_name, row in _load_latest_relax_rows(comp_dir).items():
        relax_ok = str(row.get("Relax_Success") or "").strip().upper() == "Y"
        phonon_ok = str(row.get("Phonon_Success") or "").strip().upper() == "Y"
        if not (relax_ok and phonon_ok):
            skipped.add(cif_name)

    return skipped


def step_structure_calculation(
    iteration_num: int,
    materials: list,
    n_structures: int = 5,
    max_workers: int = 1,
    relax_workers: int = 1,
    phonon_workers: int = 1,
    pressure: float = 0.0,
    device: str = "cuda",
    gpus: list = None,
    results_root: str = "results",
    seed: int | None = None,
    tracker=None,
    allow_partial_completion: bool = False,
    path_config=None,
    relax_timeout_sec: int = 120,
    prefer_isolated_relax_process: bool | None = None,
    allow_in_process_relax_fallback: bool = True,
    postprocess_workers: int = 1,
):
    """
    步骤 4：结构生成与计算。

    Args:
        iteration_num: 当前迭代轮次
        materials: 材料列表
        n_structures: 每个材料生成的结构数量
        max_workers: 结构生成并行进程数
        relax_workers: 弛豫并行进程数
        phonon_workers: 声子计算并行进程数（保留参数，当前与弛豫合并执行）
        postprocess_workers: 结构去重等 CPU 后处理的并行进程数
        pressure: 弛豫压力
        device: 计算设备
        gpus: GPU 列表；如果为 None 则使用 device 参数
        results_root: 结果存储根目录
        seed: 随机种子基准值
        tracker: 进度跟踪器实例

    Returns:
        dict: 包含计算状态的信息
    """
    print("=" * 80)
    print(f"步骤 4: 结构生成与计算 (Iteration {iteration_num})")
    print("=" * 80)

    if not materials:
        print("[ERROR] No candidate materials available for structure generation.")
        return {
            "success": False,
            "completed": False,
            "error": "No candidate materials available",
        }

    try:
        gpus = normalize_gpu_list(gpus, device=device)
    except ValueError as exc:
        print(f"[ERROR] Invalid GPU configuration: {exc}")
        return {
            "success": False,
            "completed": False,
            "error": str(exc),
        }

    SUBSTEP_GENERATION = "generation"
    SUBSTEP_RELAXATION = "relaxation"
    SUBSTEP_THERMAL = "thermal_conductivity"
    SUBSTEP_DEDUP = "deduplication"
    SUBSTEP_PHONON = "phonon_spectrum"
    relax_complete = False
    oom_encountered = False
    oom_error = None

    if path_config is not None:
        iteration_results_dir = path_config.get_iteration_results_path(iteration_num)
        gen_output_dir = iteration_results_dir / "processed_structures"
        relax_output_dir = iteration_results_dir / "MyRelaxStructure"
    else:
        iteration_results_dir = project_root / results_root / f"iteration_{iteration_num}"
        gen_output_dir = iteration_results_dir / "processed_structures"
        relax_output_dir = iteration_results_dir / "MyRelaxStructure"

    if prefer_isolated_relax_process is None:
        prefer_isolated_relax_process = os.name != "nt"

    relax_log_fieldnames = [
        "Formula",
        "Composition",
        "Material_Dir",
        "CIF_File",
        "Task_Index",
        "Relaxed_CIF",
        "Relax_Success",
        "Relax_Error",
        "Phonon_Success",
        "Phonon_Error",
        "Has_Imaginary_Freq",
        "Min_Frequency",
        "Gamma_Min_Optical",
        "Gamma_Max_Acoustic",
        "Composition_Mismatch",
        "Worker_Error_Type",
        "Worker_Exit_Code",
        "Worker_Traceback",
        "Worker_Diagnostic",
    ]

    def _append_relax_phonon_log(comp_dir: Path, row: dict):
        """Atomically append one relaxation/phonon result record."""
        log_file = comp_dir / "relax_phonon_results.csv"
        comp_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
            except (OSError, csv.Error):
                rows = []

        normalized_rows = []
        for existing_row in rows:
            composition = (
                existing_row.get("Composition")
                or existing_row.get("Material_Dir")
                or existing_row.get("Formula")
                or comp_dir.name
            )
            existing_row["Composition"] = composition
            existing_row["Material_Dir"] = existing_row.get("Material_Dir") or composition
            actual_formula = ""
            for raw_path in (
                existing_row.get("Relaxed_CIF"),
                existing_row.get("CIF_File"),
            ):
                if not raw_path:
                    continue
                candidate = Path(str(raw_path))
                if not candidate.is_absolute():
                    candidate = comp_dir / candidate
                if candidate.exists():
                    actual_formula = _actual_formula_from_relaxed_cif(candidate)
                    if actual_formula:
                        break
            existing_row["Formula"] = actual_formula or existing_row.get("Formula") or ""
            normalized_rows.append(existing_row)

        normalized_rows.append({k: row.get(k) for k in relax_log_fieldnames})
        rows = normalized_rows

        temp_file = log_file.with_name(f".{log_file.name}.{os.getpid()}.tmp")
        try:
            with open(temp_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=relax_log_fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp_file, log_file)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def _get_relaxed_cifs(formula: str):
        comp_dir = relax_output_dir / formula
        if comp_dir.exists():
            return sorted(comp_dir.glob("*.cif"))
        return []

    generation_manifest_path = gen_output_dir / "generation_status.json"
    dedup_status_path = relax_output_dir / "deduplication_status.json"
    thermal_status_path = relax_output_dir / "thermal_status.json"
    relaxation_failure_path = relax_output_dir / "relaxation_failure.json"

    def _load_relaxation_failure() -> dict[str, Any] | None:
        try:
            payload = json.loads(relaxation_failure_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_relaxation_failure(payload: dict[str, Any]) -> None:
        relaxation_failure_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = relaxation_failure_path.with_name(
            f".{relaxation_failure_path.name}.{os.getpid()}.tmp"
        )
        try:
            temp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temp_path, relaxation_failure_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _relaxation_retry_cifs() -> set[str]:
        payload = _load_relaxation_failure()
        if not payload or payload.get("status") != "incomplete":
            return set()
        values = payload.get("retry_cifs", [])
        return {Path(str(value)).name for value in values if str(value).strip()}

    def _archive_regenerated_relaxation(formula: str) -> None:
        """Preserve old relax results before a newly generated CIF is used."""
        source = relax_output_dir / formula
        if not source.exists():
            return
        archive_root = relax_output_dir / ".stale_generation_attempts" / formula
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = archive_root / f"run_{os.getpid()}_{len(list(archive_root.iterdir())) + 1}"
        shutil.copytree(source, archive)
        shutil.rmtree(source)

    def _load_generation_manifest() -> dict:
        if not generation_manifest_path.exists():
            return {}
        try:
            data = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_generation_manifest(manifest: dict) -> None:
        gen_output_dir.mkdir(parents=True, exist_ok=True)
        temp_file = generation_manifest_path.with_name(
            f".{generation_manifest_path.name}.{os.getpid()}.tmp"
        )
        try:
            temp_file.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temp_file, generation_manifest_path)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def _generated_cif_paths(formula: str) -> list[Path]:
        comp_dir = gen_output_dir / formula
        processed_dir = comp_dir / "processed"
        search_dir = processed_dir if processed_dir.exists() else comp_dir
        return sorted(search_dir.glob("*.cif")) if search_dir.exists() else []

    def _generation_status_for_count(count: int, expected: int) -> str:
        return "success_partial" if int(count) < int(expected) else "success"

    def _has_valid_generated_cifs(formula: str) -> bool:
        """Accept one or more parseable CIFs; malformed samples are not reused."""
        paths = _generated_cif_paths(formula)
        if not paths:
            return False
        try:
            from tools.structure_parallel import _filter_valid_generated_cif_outputs

            valid_paths, _ = _filter_valid_generated_cif_outputs(paths, formula)
        except Exception:
            return False
        return len(valid_paths) == len(paths) and bool(valid_paths)

    def _generation_entry_is_terminal(formula: str, manifest: dict) -> bool:
        entry = (manifest.get("materials") or {}).get(formula, {})
        if not isinstance(entry, dict):
            return _has_valid_generated_cifs(formula)
        status = entry.get("status")
        if status == "skipped":
            # Historical count-shortfall entries may contain usable partial CIFs.
            return _has_valid_generated_cifs(formula)
        # A previous backend failure is only historical when a later attempt
        # left at least one validated final CIF. Without any valid CIF it must
        # remain non-terminal and be retried/fail-fast as appropriate.
        if status not in {"success", "success_partial", "failed"}:
            return False
        return _has_valid_generated_cifs(formula)

    def _is_oom_result(result: dict) -> bool:
        error_text = " ".join(
            str(result.get(key) or "")
            for key in ("error", "relax_error", "phonon_error")
        ).lower()
        return "out of memory" in error_text or "cuda oom" in error_text

    for m in materials:
        m["formula"] = normalize_formula(str(m["formula"]))

    if tracker:
        # 仅在 substep 不存在时初始化，避免覆盖已有 completed 状态
        if not tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_GENERATION):
            # 检查 substep 是否已存在
            existing_meta = tracker.get_substep_metadata(iteration_num, "structure_calculation", SUBSTEP_GENERATION)
            if existing_meta is None:
                tracker.update_substep(
                    iteration_num,
                    "structure_calculation",
                    SUBSTEP_GENERATION,
                    {"materials_total": len(materials)},
                )
        
        if not tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_RELAXATION):
            existing_meta = tracker.get_substep_metadata(iteration_num, "structure_calculation", SUBSTEP_RELAXATION)
            if existing_meta is None:
                tracker.update_substep(
                    iteration_num,
                    "structure_calculation",
                    SUBSTEP_RELAXATION,
                    {"structures_total": 0},
                )
        
        if not tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_PHONON):
            existing_meta = tracker.get_substep_metadata(iteration_num, "structure_calculation", SUBSTEP_PHONON)
            if existing_meta is None:
                tracker.update_substep(
                    iteration_num,
                    "structure_calculation",
                    SUBSTEP_PHONON,
                    {"note": "combined_with_relaxation"},
                )

    # === 4.1 生成结构 ===
    gen_output_dir.mkdir(parents=True, exist_ok=True)
    relax_output_dir.mkdir(parents=True, exist_ok=True)
    generation_manifest = _load_generation_manifest()
    generation_records = generation_manifest.setdefault("materials", {})
    materials_original = list(materials)
    materials_existing: list[str] = []
    materials_skipped: list[str] = []
    generated_formulas: list[str] = []
    skipped_details: dict[str, Any] = {}

    generation_marked = tracker and tracker.is_substep_completed(
        iteration_num, "structure_calculation", SUBSTEP_GENERATION
    )
    if generation_marked:
        missing = [
            m["formula"]
            for m in materials_original
            if not _generation_entry_is_terminal(m["formula"], generation_manifest)
        ]
        if missing:
            print(f"[WARN] 结构生成进度记录已完成，但产物无效或缺失: {missing}")
            if tracker:
                tracker.reset_substep(iteration_num, "structure_calculation", SUBSTEP_GENERATION)
            generation_marked = False

    if generation_marked:
        print("\n[SKIP] 子步骤 4.1（生成结构）已完成，跳过")
        for material in materials_original:
            formula = material["formula"]
            entry = generation_records.get(formula, {})
            if isinstance(entry, dict) and entry.get("status") == "skipped":
                if _has_valid_generated_cifs(formula):
                    paths = _generated_cif_paths(formula)
                    generation_records[formula] = {
                        **entry,
                        "status": _generation_status_for_count(len(paths), n_structures),
                        "generator_backend": entry.get("generator_backend") or "CrystaLLM",
                        "cif_files": [path.name for path in paths],
                        "n_structures": len(paths),
                        "requested_structures": n_structures,
                        "partial": len(paths) < n_structures,
                        "source": "validated_partial_existing",
                    }
                    materials_existing.append(formula)
                else:
                    materials_skipped.append(formula)
                    skipped_details[formula] = entry
            elif _has_valid_generated_cifs(formula):
                materials_existing.append(formula)
    else:
        print(f"\n{'=' * 80}")
        print("4.1 生成晶体结构")
        print(f"{'=' * 80}")

        materials_to_gen = []
        known_valid_formulas = {
            str(item["formula"])
            for item in materials_original
            if _has_valid_generated_cifs(str(item["formula"]))
        }
        partial_batch_has_valid_output = bool(known_valid_formulas)
        for material in materials_original:
            formula = material["formula"]
            entry = generation_records.get(formula, {})
            if isinstance(entry, dict) and entry.get("status") == "skipped":
                if _has_valid_generated_cifs(formula):
                    paths = _generated_cif_paths(formula)
                    generation_records[formula] = {
                        **entry,
                        "status": _generation_status_for_count(len(paths), n_structures),
                        "generator_backend": entry.get("generator_backend") or "CrystaLLM",
                        "cif_files": [path.name for path in paths],
                        "n_structures": len(paths),
                        "requested_structures": n_structures,
                        "partial": len(paths) < n_structures,
                        "source": "validated_partial_existing",
                    }
                    print(f"  [SKIP] {formula}: 发现历史 partial 有效结构，恢复使用 {len(paths)}/{n_structures}")
                    materials_existing.append(formula)
                else:
                    print(f"  [SKIP] {formula}: 已记录为结构失败，跳过重试")
                    materials_skipped.append(formula)
                    skipped_details[formula] = entry
            elif (
                isinstance(entry, dict)
                and entry.get("status") == "failed"
                and partial_batch_has_valid_output
                and formula not in known_valid_formulas
            ):
                # A previous run may have stopped on this material after other
                # materials in the same batch succeeded. Preserve the failure
                # as audit history, but do not retry it on every resume.
                normalized_entry = {
                    **entry,
                    "status": "skipped",
                    "reason": "material_generation_failed_after_partial_batch_success",
                }
                generation_records[formula] = normalized_entry
                print(f"  [SKIP] {formula}: 历史单材料生成失败，已有同批次有效结构，跳过重试")
                materials_skipped.append(formula)
                skipped_details[formula] = normalized_entry
            elif _has_valid_generated_cifs(formula):
                print(f"  [SKIP] {formula}: 已存在通过质量检查的结构")
                materials_existing.append(formula)
            else:
                materials_to_gen.append(material)

        if tracker:
            tracker.update_substep(
                iteration_num,
                "structure_calculation",
                SUBSTEP_GENERATION,
                {
                    "materials_total": len(materials_original),
                    "materials_existing": materials_existing,
                    "materials_skipped": materials_skipped,
                    "materials_pending": [m["formula"] for m in materials_to_gen],
                },
            )

        gen_results: list[dict] = []
        if materials_to_gen:
            print(f"[INFO] 准备为 {len(materials_to_gen)} 个材料生成结构...")
            gen_results = generate_structures_parallel(
                materials=materials_to_gen,
                device=device,
                output_dir=str(gen_output_dir),
                n_structures=n_structures,
                relax_structures=False,
                max_workers=max_workers,
                gpus=gpus,
                seed=seed,
            )

        success_results = [r for r in gen_results if r.get("success")]
        skipped_results = [
            r for r in gen_results
            if not r.get("success") and (r.get("skipped") or r.get("status") == "skipped")
        ]
        generation_failures = [
            r for r in gen_results
            if not r.get("success") and r not in skipped_results
        ]
        # A single material-level backend/runtime failure must not block a
        # batch when other materials produced valid structures. If the whole
        # batch failed (and no previously valid material exists), preserve the
        # fail-fast behavior for a genuinely unavailable CrystaLLM backend.
        global_failures = generation_failures if (
            generation_failures and not success_results and not materials_existing
        ) else []
        recoverable_failures = [r for r in generation_failures if r not in global_failures]
        for result in recoverable_failures:
            skipped_results.append({
                **result,
                "success": False,
                "status": "skipped",
                "skipped": True,
                "fatal": False,
                "skip_reason": "material_generation_failed_after_partial_batch_success",
            })
        generated_formulas = [str(r.get("formula")) for r in success_results]
        success_count = len(success_results)
        print(f"[OK] 结构生成完成！成功: {success_count}/{len(materials_to_gen)}")

        for result in skipped_results:
            formula = str(result.get("formula") or "Unknown")
            materials_skipped.append(formula)
            skipped_details[formula] = result

        for result in success_results:
            formula = str(result.get("formula") or "Unknown")
            requested_count = int(result.get("requested_structures", n_structures))
            actual_count = int(
                result.get("n_structures")
                or len(result.get("cif_files", []))
                or len(_generated_cif_paths(formula))
            )
            generation_records[formula] = {
                "status": result.get("status") or _generation_status_for_count(actual_count, requested_count),
                "generator_backend": result.get("generator_backend"),
                "seed": result.get("seed"),
                "cif_files": result.get("cif_files", []),
                "n_structures": actual_count,
                "requested_structures": requested_count,
                "partial": bool(result.get("partial")) or actual_count < requested_count,
                "precheck_status": result.get("precheck_status"),
                "parseable_cif_count": int(result.get("parseable_cif_count") or actual_count),
                "parse_failed_cif_count": int(result.get("parse_failed_cif_count") or 0),
                "rejected_cif_count": int(result.get("rejected_cif_count") or 0),
                "validation_errors": result.get("validation_errors", []),
                "attempts": result.get("attempts", []),
            }
        for result in skipped_results:
            formula = str(result.get("formula") or "Unknown")
            generation_records[formula] = {
                "status": "skipped",
                "reason": result.get("skip_reason", "structure_invalid_after_seed_retry"),
                "error": result.get("error"),
                "attempts": result.get("attempts", []),
            }
        for result in global_failures:
            formula = str(result.get("formula") or "Unknown")
            generation_records[formula] = {
                "status": "failed",
                "reason": "backend_failure",
                "error": result.get("error"),
                "attempts": result.get("attempts", []),
            }
        generation_manifest.update(
            {
                "iteration": iteration_num,
                "n_structures": n_structures,
                "materials_total": len(materials_original),
                "status": (
                    "failed"
                    if global_failures
                    else "completed_with_partial_results"
                    if skipped_results
                    or any(bool(result.get("partial")) for result in success_results)
                    else "completed"
                ),
                "materials_skipped": sorted(
                    str(result.get("formula") or "Unknown")
                    for result in skipped_results
                ),
                "materials_failed": sorted(
                    str(result.get("formula") or "Unknown")
                    for result in global_failures
                ),
            }
        )
        _save_generation_manifest(generation_manifest)

        if global_failures:
            failed_formulas = [str(r.get("formula") or "Unknown") for r in global_failures]
            print(
                "[ERROR] CrystaLLM backend unavailable for the complete generation batch; "
                "no fallback structure is used. Failed materials: "
                f"{failed_formulas}"
            )
            if tracker:
                tracker.update_substep(
                    iteration_num,
                    "structure_calculation",
                    SUBSTEP_GENERATION,
                    {
                        "materials_generated": generated_formulas,
                        "materials_existing": materials_existing,
                        "materials_skipped": materials_skipped,
                        "materials_failed": failed_formulas,
                        "skipped_details": skipped_details,
                        "fallback_used": False,
                    },
                    completed=False,
                )
            safe_clear_memory(device)
            return {
                "success": False,
                "completed": False,
                "error": "CrystaLLM backend failure during structure generation",
                "failed_formulas": failed_formulas,
                "skipped_formulas": materials_skipped,
                "gen_output_dir": str(gen_output_dir),
                "relax_output_dir": str(relax_output_dir),
            }

        if recoverable_failures:
            skipped_formulas = [str(r.get("formula") or "Unknown") for r in recoverable_failures]
            print(
                "[WARN] 单材料 CrystaLLM 生成失败，跳过当前材料并继续其余材料: "
                f"{skipped_formulas}"
            )
        if skipped_results:
            print(
                "[WARN] 部分结构在首次生成和 seed 重试后仍无效，"
                f"跳过当前材料: {materials_skipped}"
            )
        elif not materials_to_gen:
            print("[OK] 所有结构已生成，检测到后跳过")

        if tracker:
            tracker.update_substep(
                iteration_num,
                "structure_calculation",
                SUBSTEP_GENERATION,
                {
                    "status": generation_manifest.get("status", "completed"),
                    "materials_generated": generated_formulas,
                    "materials_existing": materials_existing,
                    "materials_skipped": materials_skipped,
                    "materials_partial": generation_manifest.get("materials_partial", []),
                    "skipped_details": skipped_details,
                    "materials_failed": [],
                    "fallback_used": False,
                },
            )
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_GENERATION,
                {
                    "status": generation_manifest.get("status", "completed"),
                    "materials_processed": len(materials_original),
                    "materials_partial": len(generation_manifest.get("materials_partial", [])),
                    "materials_skipped": len(materials_skipped),
                },
            )
        safe_clear_memory(device)

    for formula in materials_existing:
        existing_entry = generation_records.get(formula)
        if isinstance(existing_entry, dict) and existing_entry.get("status") == "failed":
            existing_paths = _generated_cif_paths(formula)
            generation_records[formula] = {
                **existing_entry,
                "status": _generation_status_for_count(len(existing_paths), n_structures),
                "generator_backend": existing_entry.get("generator_backend") or "CrystaLLM",
                "cif_files": [path.name for path in existing_paths],
                "n_structures": len(existing_paths),
                "requested_structures": n_structures,
                "partial": len(existing_paths) < n_structures,
                "source": "validated_existing",
            }
        else:
            generation_records.setdefault(
                formula,
                {
                    "status": "success",
                    "generator_backend": "CrystaLLM",
                    "cif_files": [path.name for path in _generated_cif_paths(formula)],
                    "source": "validated_existing",
                },
            )
    for formula, entry in generation_records.items():
        if not isinstance(entry, dict) or entry.get("status") not in {"success", "success_partial"}:
            continue
        actual_count = int(
            entry.get("n_structures")
            or len(entry.get("cif_files", []))
            or len(_generated_cif_paths(formula))
        )
        requested_count = int(entry.get("requested_structures", n_structures))
        entry["status"] = _generation_status_for_count(actual_count, requested_count)
        entry["n_structures"] = actual_count
        entry["requested_structures"] = requested_count
        entry["partial"] = actual_count < requested_count

    generation_entries = [
        entry for entry in generation_records.values()
        if isinstance(entry, dict)
    ]
    generation_partial_formulas = sorted(
        formula
        for formula, entry in generation_records.items()
        if isinstance(entry, dict) and entry.get("status") == "success_partial"
    )
    generation_skipped_formulas = sorted(
        formula
        for formula, entry in generation_records.items()
        if isinstance(entry, dict) and entry.get("status") == "skipped"
    )
    generation_failed_formulas = sorted(
        formula
        for formula, entry in generation_records.items()
        if isinstance(entry, dict) and entry.get("status") in {"failed", "fatal"}
    )
    generation_manifest.update(
        {
            "iteration": iteration_num,
            "n_structures": n_structures,
            "materials_total": len(materials_original),
            "status": (
                "completed_with_partial_results"
                if generation_partial_formulas
                or generation_skipped_formulas
                or generation_failed_formulas
                else "completed"
            ),
            "materials_success": sum(
                1 for entry in generation_entries
                if entry.get("status") in {"success", "success_partial"}
            ),
            "materials_partial": generation_partial_formulas,
            "materials_skipped": generation_skipped_formulas,
            "materials_failed": generation_failed_formulas,
            "structure_counts": {
                formula: {
                    "requested": int(entry.get("requested_structures", n_structures)),
                    "accepted": int(entry.get("n_structures") or len(entry.get("cif_files", []))),
                    "precheck_status": entry.get("precheck_status"),
                    "parse_failed": int(entry.get("parse_failed_cif_count") or 0),
                    "status": entry.get("status"),
                }
                for formula, entry in generation_records.items()
                if isinstance(entry, dict)
            },
        }
    )
    _save_generation_manifest(generation_manifest)
    if tracker:
        tracker.update_substep(
            iteration_num,
            "structure_calculation",
            SUBSTEP_GENERATION,
            {
                "status": generation_manifest.get("status", "completed"),
                "materials_success": generation_manifest.get("materials_success", 0),
                "materials_partial": generation_manifest.get("materials_partial", []),
                "materials_skipped": generation_manifest.get("materials_skipped", []),
                "materials_failed": generation_manifest.get("materials_failed", []),
                "structure_counts": generation_manifest.get("structure_counts", {}),
            },
        )

    for formula in generated_formulas:
        _archive_regenerated_relaxation(formula)

    active_formulas = set(materials_existing) | set(generated_formulas)
    materials = [
        material for material in materials_original
        if material["formula"] in active_formulas
    ]
    if not materials:
        error = "No valid generated structures available for relaxation"
        print(f"[ERROR] {error}; stopping this iteration")
        if tracker:
            tracker.update_substep(
                iteration_num,
                "structure_calculation",
                SUBSTEP_RELAXATION,
                {"systemic_failure": True, "error": error},
                completed=False,
            )
            tracker.update_substep(
                iteration_num,
                "structure_calculation",
                SUBSTEP_PHONON,
                {"systemic_failure": True, "error": error},
                completed=False,
            )
        return {
            "success": False,
            "completed": False,
            "error": error,
            "systemic_failure": True,
            "gen_output_dir": str(gen_output_dir),
            "relax_output_dir": str(relax_output_dir),
            "materials_generated": generated_formulas,
            "materials_skipped": materials_skipped,
            "skipped_details": skipped_details,
            "no_valid_structures": True,
        }

    # === 4.2 弛豫 + 声子计算 ===
    relax_stage_status = "completed"
    relax_partial = False
    relax_marked = tracker and tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_RELAXATION)
    failure_state = _load_relaxation_failure()
    recovered_from_failure = bool(
        failure_state and failure_state.get("status") == "incomplete"
    )
    retry_cifs = _relaxation_retry_cifs()
    if relax_marked:
        relax_metadata = tracker.get_substep_metadata(
            iteration_num, "structure_calculation", SUBSTEP_RELAXATION
        ) or {}
        partial_relax_marked = bool(relax_metadata.get("partial"))
        relax_partial = partial_relax_marked
        relax_stage_status = str(relax_metadata.get("status") or "completed")
        relax_artifacts_valid = True
        completed_any = False
        if failure_state and failure_state.get("status") == "incomplete":
            relax_artifacts_valid = False
        for m in materials:
            formula = m["formula"]
            gen_comp_dir = gen_output_dir / formula
            search_dir = gen_comp_dir / "processed" if (gen_comp_dir / "processed").exists() else gen_comp_dir
            source_cifs = {p.name for p in search_dir.glob("*.cif")} if search_dir.exists() else set()
            if not source_cifs:
                relax_artifacts_valid = False
                break
            relax_comp_dir = relax_output_dir / formula
            completed_cifs = _load_completed_phonon_cifs(relax_comp_dir)
            terminal_skipped_cifs = _load_terminal_skipped_cifs(relax_comp_dir)
            # 成功完成 relaxation+phonon，或已有明确失败结果的结构，
            # 都视为本结构的 terminal 状态，避免单结构异常阻塞整轮。
            if completed_cifs:
                completed_any = True
            if not source_cifs.issubset(completed_cifs | terminal_skipped_cifs):
                relax_artifacts_valid = False
                break
        persisted_terminal_count = sum(
            len(_load_terminal_skipped_cifs(relax_output_dir / m["formula"]))
            for m in materials
        )
        if persisted_terminal_count:
            relax_partial = True
            relax_stage_status = "completed_with_partial_results"
        if materials and not completed_any:
            # A round with only failed/terminal structures has no valid
            # downstream input and must not be resumed as completed.
            relax_artifacts_valid = False

        if not relax_artifacts_valid:
            print("\n[WARN] 子步骤 4.2 进度已记录完成，但弛豫产物不完整，重置后重跑")
            if tracker:
                tracker.reset_substep(iteration_num, "structure_calculation", SUBSTEP_RELAXATION)
                tracker.reset_substep(iteration_num, "structure_calculation", SUBSTEP_PHONON)
                tracker.reset_substep(iteration_num, "structure_calculation", SUBSTEP_DEDUP)
                tracker.reset_substep(iteration_num, "structure_calculation", SUBSTEP_THERMAL)
            relax_marked = False

    if relax_marked:
        print("\n[SKIP] 子步骤 4.2（弛豫 + 声子计算）已完成，跳过")
        relax_complete = True
    else:
        print(f"\n{'=' * 80}")
        print("4.2 Relax structures + phonon calculation")
        print(f"{'=' * 80}")

        mode_label = "isolated subprocess" if prefer_isolated_relax_process else "in-process"
        print(f"  [INFO] Relax worker mode: {mode_label}")

        relax_tasks = []
        task_idx = 0
        materials_with_structures = []
        source_cifs_map = {}
        completed_cifs_by_formula: dict[str, set[str]] = {}
        terminal_cifs_by_formula: dict[str, set[str]] = {}

        # Older interrupted runs may contain only terminal failure rows but no
        # relaxation_failure.json.  Detect that legacy all-failed state before
        # applying the per-structure terminal-skip rule, so the next run can
        # retry the full batch after the worker/runtime is repaired.
        for m in materials:
            formula = m["formula"]
            gen_comp_dir = gen_output_dir / formula
            relax_comp_dir = relax_output_dir / formula
            processed_dir = gen_comp_dir / "processed"
            search_dir = processed_dir if processed_dir.exists() else gen_comp_dir
            existing_cifs = sorted(search_dir.glob("*.cif")) if search_dir.exists() else []
            source_cifs_map[formula] = {c.name for c in existing_cifs}
            completed_cifs_by_formula[formula] = _load_completed_phonon_cifs(relax_comp_dir)
            terminal_cifs_by_formula[formula] = _load_terminal_skipped_cifs(relax_comp_dir)

        if not any(completed_cifs_by_formula.values()):
            for terminal_cifs in terminal_cifs_by_formula.values():
                retry_cifs.update(terminal_cifs)

        for m in materials:
            formula = m["formula"]
            gen_comp_dir = gen_output_dir / formula
            relax_comp_dir = relax_output_dir / formula
            processed_dir = gen_comp_dir / "processed"
            attempted_cifs, _ = _load_relax_status(relax_comp_dir)
            completed_phonon_cifs = completed_cifs_by_formula.get(formula, set())
            terminal_skipped_cifs = terminal_cifs_by_formula.get(formula, set())

            search_dir = processed_dir if processed_dir.exists() else gen_comp_dir

            if search_dir.exists():
                existing_cifs = sorted(search_dir.glob("*.cif"))
                if len(existing_cifs) >= n_structures:
                    materials_with_structures.append(formula)
                for cif_file in existing_cifs:
                    # A task is resumable until its phonon result is persisted.
                    # A relaxed CIF alone is not enough: the process may have
                    # been interrupted between relaxation and phonon analysis.
                    if cif_file.name in completed_phonon_cifs:
                        continue
                    # Any persisted non-success result is a terminal skip for
                    # this structure; systemic batch failures are explicitly
                    # listed in retry_cifs so a corrected worker can resume.
                    if cif_file.name in terminal_skipped_cifs and cif_file.name not in retry_cifs:
                        continue
                    if (
                        allow_partial_completion
                        and cif_file.name in attempted_cifs
                        and cif_file.name not in retry_cifs
                    ):
                        continue
                    assigned_gpu = gpus[task_idx % len(gpus)]
                    task_seed = _derive_seed(seed, "relax", iteration_num, formula, cif_file.name)
                    relax_tasks.append((str(cif_file), formula, str(relax_output_dir), pressure, assigned_gpu, task_seed))
                    task_idx += 1

        if tracker and not tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_GENERATION):
            if len(materials_with_structures) == len(materials):
                tracker.mark_substep_completed(
                    iteration_num,
                    "structure_calculation",
                    SUBSTEP_GENERATION,
                    {"materials_processed": len(materials)},
                )

        if tracker:
            materials_progress = {}
            for m in materials:
                formula = m["formula"]
                total_structures = len(
                    sorted((gen_output_dir / formula / "processed").glob("*.cif"))
                    if (gen_output_dir / formula / "processed").exists()
                    else sorted((gen_output_dir / formula).glob("*.cif"))
                )
                materials_progress[formula] = {
                    "total": total_structures,
                    "processed": 0,
                    "relax_success": 0,
                    "phonon_success": 0,
                }
            tracker.update_substep(
                iteration_num,
                "structure_calculation",
                SUBSTEP_RELAXATION,
                {
                    "structures_total": len(relax_tasks),
                    "materials_progress": materials_progress,
                },
            )

        relaxed_count = 0
        phonon_success_count = 0
        processed_count = 0
        oom_encountered = False
        oom_count = 0
        oom_error = None

        if relax_tasks:
            workers_per_gpu = max(1, int(relax_workers)) if len(gpus) > 1 else 1
            actual_workers = len(gpus) * workers_per_gpu
            print(f"Total structures to relax: {len(relax_tasks)}")
            print(f"  - GPU数量: {len(gpus)}")
            print(f"  - 每 GPU 并行数: {workers_per_gpu}")
            print(f"  - 总并行数: {actual_workers}")
            print("  - GPU 任务分配:")
            gpu_task_count = {gpu: 0 for gpu in gpus}
            for task in relax_tasks:
                gpu_task_count[task[4]] += 1
            for gpu, count in gpu_task_count.items():
                print(f"      {gpu}: {count} tasks")

            materials_progress = {}
            if tracker:
                current_meta = tracker.get_substep_metadata(
                    iteration_num, "structure_calculation", SUBSTEP_RELAXATION
                ) or {}
                materials_progress = current_meta.get("materials_progress", {})

            def _run_relax(task):
                try:
                    return run_relax_task_with_timeout(
                        task,
                        relax_timeout_sec,
                        prefer_subprocess=prefer_isolated_relax_process,
                        allow_in_process_fallback=allow_in_process_relax_fallback,
                    )
                except BaseException as exc:
                    import traceback

                    diagnostic_path = (
                        Path(task[2])
                        / str(task[1])
                        / "worker_diagnostics"
                        / f"{Path(task[0]).stem}.worker.log"
                    )
                    return {
                        "success": False,
                        "formula": None,
                        "composition": task[1],
                        "cif_file": Path(task[0]).name,
                        "error": str(exc),
                        "worker_error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                        "worker_diagnostic": str(diagnostic_path),
                    }

            def _persist_relax_result(result_index: int, task: tuple, res: dict) -> None:
                """Persist one completed worker result from the parent process."""
                composition = res.get("composition") or task[1]
                actual_formula = res.get("formula") or res.get("relaxed_formula")
                log_row = {
                    "Formula": actual_formula,
                    "Composition": composition,
                    "Material_Dir": composition,
                    "CIF_File": res.get("cif_file") or Path(task[0]).name,
                    "Task_Index": result_index + 1,
                    "Relaxed_CIF": res.get("file"),
                    "Relax_Success": "Y" if res.get("success") else "N",
                    "Relax_Error": res.get("relax_error") or res.get("error"),
                    "Phonon_Success": "Y" if res.get("phonon_success") else "N",
                    "Phonon_Error": res.get("phonon_error"),
                    "Has_Imaginary_Freq": res.get("has_imaginary"),
                    "Min_Frequency": res.get("min_frequency"),
                    "Gamma_Min_Optical": res.get("gamma_min_optical"),
                    "Gamma_Max_Acoustic": res.get("gamma_max_acoustic"),
                    "Composition_Mismatch": "Y" if res.get("composition_mismatch") else "N",
                    "Worker_Error_Type": res.get("worker_error_type") or res.get("error_type"),
                    "Worker_Exit_Code": res.get("worker_exit_code"),
                    "Worker_Traceback": res.get("traceback"),
                    "Worker_Diagnostic": res.get("worker_diagnostic"),
                }
                _append_relax_phonon_log(relax_output_dir / composition, log_row)

            def _restore_relax_log_order() -> None:
                """Normalize checkpoint rows to the original task order."""
                for composition in materials_progress:
                    log_file = relax_output_dir / composition / "relax_phonon_results.csv"
                    if not log_file.exists():
                        continue
                    try:
                        with open(log_file, "r", encoding="utf-8-sig", newline="") as handle:
                            rows = list(csv.DictReader(handle))
                        rows.sort(
                            key=lambda row: (
                                int(row.get("Task_Index"))
                                if str(row.get("Task_Index") or "").strip().isdigit()
                                else 10**9
                            )
                        )
                        temp_file = log_file.with_name(f".{log_file.name}.{os.getpid()}.tmp")
                        with open(temp_file, "w", newline="", encoding="utf-8-sig") as handle:
                            writer = csv.DictWriter(handle, fieldnames=relax_log_fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)
                        os.replace(temp_file, log_file)
                    except (OSError, csv.Error, ValueError):
                        continue

            relax_results = run_tasks_by_gpu(
                relax_tasks,
                gpus=gpus,
                workers_per_gpu=workers_per_gpu,
                task_device=lambda task: task[4],
                run_task=_run_relax,
                on_result=_persist_relax_result,
            )

            # The scheduler returns input order, so counters and final
            # reporting remain deterministic even though checkpoint callbacks
            # persist rows as individual workers complete.
            for i, (task, res) in enumerate(zip(relax_tasks, relax_results)):
                processed_count += 1
                composition = res.get("composition") or task[1]
                actual_formula = res.get("formula") or res.get("relaxed_formula")
                if composition in materials_progress:
                    materials_progress[composition]["processed"] += 1

                relax_success = bool(res.get("success"))
                if relax_success:
                    relaxed_count += 1
                    if composition in materials_progress:
                        materials_progress[composition]["relax_success"] += 1

                phonon_success = bool(res.get("phonon_success"))
                if phonon_success:
                    phonon_success_count += 1
                    if composition in materials_progress:
                        materials_progress[composition]["phonon_success"] += 1

                assigned_gpu = task[4]
                status = "completed" if relax_success else f"failed: {res.get('error')}"
                print(
                    f"  [{i + 1}/{len(relax_tasks)}] {composition} "
                    f"({assigned_gpu}) Relaxation {status}"
                )

                if tracker:
                    tracker.update_substep(
                        iteration_num,
                        "structure_calculation",
                        SUBSTEP_RELAXATION,
                        {
                            "structures_processed": processed_count,
                            "relax_success": relaxed_count,
                            "phonon_success": phonon_success_count,
                            "last_formula": actual_formula,
                            "last_composition": composition,
                            "materials_progress": materials_progress,
                        },
                    )

                if _is_oom_result(res):
                    oom_encountered = True
                    oom_count += 1
                    oom_error = res.get("error") or res.get("relax_error") or res.get("phonon_error")
                    print(
                        "  [WARN] MatterSim CUDA OOM detected; keeping any current "
                        "result, skipping this structure, and continuing with the next."
                    )

            _restore_relax_log_order()
            safe_clear_memory(device)
            if oom_encountered:
                print(
                    f"[WARN] Relaxation encountered {oom_count} CUDA OOM event(s); "
                    f"skipped those current structure(s), then continued."
                )
            print(f"弛豫完成！成功: {relaxed_count}/{len(relax_tasks)}")

        completed_cif_count = sum(
            len(_load_completed_phonon_cifs(relax_output_dir / formula))
            for formula in source_cifs_map
        )
        attempted_failure_rows = [
            {
                "formula": res.get("composition") or task[1],
                "cif_file": res.get("cif_file") or Path(task[0]).name,
                "error": res.get("error") or res.get("relax_error") or "unknown worker failure",
                "error_type": res.get("worker_error_type") or res.get("error_type"),
                "exit_code": res.get("worker_exit_code"),
                "traceback": res.get("traceback"),
                "worker_diagnostic": res.get("worker_diagnostic"),
            }
            for task, res in zip(relax_tasks, relax_results)
            if not res.get("success")
        ] if relax_tasks else []

        if materials and completed_cif_count == 0:
            # Per-structure skips remain legal when another structure yields a
            # valid result.  A batch with no valid relaxation+phonon artifact,
            # however, is a systemic/incomplete iteration and must be retried
            # before deduplication, extraction, or the next iteration.
            retry_cifs = sorted(
                {
                    str(row.get("cif_file"))
                    for row in attempted_failure_rows
                    if str(row.get("cif_file") or "").strip()
                }
            )
            failure_payload = {
                "iteration": iteration_num,
                "status": "incomplete",
                "reason": "no_valid_relaxation_phonon_artifact",
                "structures_attempted": len(relax_tasks),
                "structures_failed": len(attempted_failure_rows),
                "retry_cifs": retry_cifs,
                "failures": attempted_failure_rows,
            }
            _write_relaxation_failure(failure_payload)
            if tracker:
                tracker.update_substep(
                    iteration_num,
                    "structure_calculation",
                    SUBSTEP_RELAXATION,
                    {
                        "completed": False,
                        "systemic_failure": True,
                        "failure_status": str(relaxation_failure_path),
                        "structures_processed": processed_count,
                        "relax_success": relaxed_count,
                        "phonon_success": phonon_success_count,
                        "failures": attempted_failure_rows,
                    },
                    completed=False,
                )
                tracker.update_substep(
                    iteration_num,
                    "structure_calculation",
                    SUBSTEP_PHONON,
                    {
                        "completed": False,
                        "systemic_failure": True,
                        "failure_status": str(relaxation_failure_path),
                    },
                    completed=False,
                )
            print(
                "[ERROR] Relaxation/phonon produced no valid artifact; "
                "stopping this iteration before deduplication and thermal calculation."
            )
            return {
                "success": False,
                "completed": False,
                "error": "No valid relaxation+phonon artifacts; iteration is incomplete",
                "systemic_failure": True,
                "failure_status": str(relaxation_failure_path),
                "gen_output_dir": str(gen_output_dir),
                "relax_output_dir": str(relax_output_dir),
                "materials_generated": generated_formulas,
                "materials_skipped": materials_skipped,
                "skipped_details": skipped_details,
                "no_valid_structures": False,
                "relaxation_failures": attempted_failure_rows,
            }

        if failure_state and failure_state.get("status") == "incomplete":
            _write_relaxation_failure(
                {
                    **failure_state,
                    "status": "recovered",
                    "recovered_iteration": iteration_num,
                    "recovered_valid_relaxation_cifs": completed_cif_count,
                }
            )

        relax_complete = True
        for m in materials:
            formula = m["formula"]
            src_cifs = source_cifs_map.get(formula, set())
            if not src_cifs:
                relax_complete = False
                break
            comp_dir = relax_output_dir / formula
            completed_cifs = _load_completed_phonon_cifs(comp_dir)
            terminal_skipped_cifs = _load_terminal_skipped_cifs(comp_dir)
            done = completed_cifs | terminal_skipped_cifs
            if not src_cifs.issubset(done):
                relax_complete = False
                break

        source_structure_count = sum(len(cifs) for cifs in source_cifs_map.values())
        persisted_completed_count = sum(
            len(_load_completed_phonon_cifs(relax_output_dir / formula))
            for formula in source_cifs_map
        )
        persisted_terminal_count = sum(
            len(_load_terminal_skipped_cifs(relax_output_dir / formula))
            for formula in source_cifs_map
        )
        relax_partial = persisted_terminal_count > 0
        relax_stage_status = (
            "completed_with_partial_results" if relax_partial else "completed"
        )

        if tracker and relax_complete:
            relax_metadata = {
                "status": relax_stage_status,
                "structures_total": source_structure_count,
                "structures_attempted": processed_count,
                "structures_completed": persisted_completed_count,
                "structures_failed": persisted_terminal_count,
                "structures_relaxed": relaxed_count if relax_tasks else 0,
                "structures_processed": processed_count,
                "relax_success": relaxed_count if relax_tasks else 0,
                "phonon_success": phonon_success_count if relax_tasks else 0,
                "oom_events": oom_count,
                "oom_structures_skipped": oom_count,
                "partial": relax_partial,
            }
            if oom_error:
                relax_metadata["last_oom_error"] = str(oom_error)
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_RELAXATION,
                relax_metadata,
            )
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_PHONON,
                {
                    "status": relax_stage_status,
                    "combined_with_relaxation": True,
                    "structures_total": source_structure_count,
                    "structures_completed": persisted_completed_count,
                    "structures_failed": persisted_terminal_count,
                    "oom_events": oom_count,
                    "partial": relax_partial,
                },
            )
        safe_clear_memory(device)

    # === 4.3 弛豫结构去重（避免重复计算） ===
    dedup_success = False
    dedup_reconciled = False
    if tracker and not tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_DEDUP):
        try:
            dedup_status = json.loads(dedup_status_path.read_text(encoding="utf-8"))
            expected_formulas = sorted(m["formula"] for m in materials)
            recorded_formulas = sorted(dedup_status.get("formulas", [])) if isinstance(dedup_status, dict) else []
            dedup_reconciled = (
                not recovered_from_failure
                and isinstance(dedup_status, dict)
                and dedup_status.get("status") == "completed"
                and recorded_formulas == expected_formulas
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            dedup_reconciled = False
        if dedup_reconciled:
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_DEDUP,
                {"reconciled": True, "status_file": str(dedup_status_path)},
            )
            dedup_success = True
            print("\n[RESUME] 子步骤 4.3 已有完整状态文件，补记完成并跳过")

    if tracker and tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_DEDUP):
        print("\n[SKIP] 子步骤 4.3（结构去重）已完成，跳过")
        dedup_success = True
    elif dedup_reconciled:
        pass
    else:
        print(f"\n{'=' * 80}")
        print("4.3 弛豫结构去重")
        print(f"{'=' * 80}")

        try:
            from tools.structure_deduplicator import deduplicate_relaxed_structures

            formulas = [m["formula"] for m in materials]
            dedup_ltol = 0.2
            dedup_stol = 0.3
            dedup_angle_tol = 5.0
            dedup_attempt_supercell = True
            print(
                f"  matcher params: ltol={dedup_ltol}, stol={dedup_stol}, "
                f"angle_tol={dedup_angle_tol}, attempt_supercell={dedup_attempt_supercell}"
            )
            dedup_results = deduplicate_relaxed_structures(
                relax_dir=relax_output_dir,
                formulas=formulas,
                keep_duplicates=False,
                ltol=dedup_ltol,
                stol=dedup_stol,
                angle_tol=dedup_angle_tol,
                attempt_supercell=dedup_attempt_supercell,
                max_workers=max(1, int(postprocess_workers)),
            )

            total_removed = 0
            csv_sync_count = 0

            for formula, result in dedup_results.items():
                csv_status = "CSV已同步" if result.get("csv_updated") else ""
                print(
                    f"  {formula}: {result['total']} -> {result['unique']} "
                    f"(删除 {result['removed']}) {csv_status}".rstrip()
                )
                if result["removed"] > 0:
                    total_removed += result["removed"]
                    if result.get("csv_updated"):
                        csv_sync_count += 1

            if total_removed > 0:
                print(f"[OK] Dedup complete: removed {total_removed} duplicates, synced {csv_sync_count} CSV files")
            else:
                print("[OK] 无重复结构，跳过")

            dedup_status_payload = {
                "iteration": iteration_num,
                "status": "completed",
                "formulas": sorted(formulas),
                "materials_deduplicated": len(materials),
                "results": {
                    formula: {
                        "unique_files": [Path(path).name for path in result.get("unique_files", [])],
                        "duplicate_files": [Path(path).name for path in result.get("duplicate_files", [])],
                    }
                    for formula, result in dedup_results.items()
                },
            }
            temp_status_path = dedup_status_path.with_name(
                f".{dedup_status_path.name}.{os.getpid()}.tmp"
            )
            try:
                temp_status_path.write_text(
                    json.dumps(dedup_status_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                os.replace(temp_status_path, dedup_status_path)
            finally:
                if temp_status_path.exists():
                    try:
                        temp_status_path.unlink()
                    except OSError:
                        pass
            dedup_success = True

        except Exception as e:
            print(f"WARNING: Structure deduplication failed: {e}")
            print("Continuing with subsequent calculations...")

        if tracker and relax_complete and dedup_success:
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_DEDUP,
                {"materials_deduplicated": len(materials)},
            )
        safe_clear_memory(device)

    # === 4.4 计算热导率 ===
    def _valid_thermal_materials() -> set[str]:
        valid: set[str] = set()
        for material in materials:
            formula = material["formula"]
            comp_dir = relax_output_dir / formula
            expected_cifs = _load_completed_phonon_relaxed_cifs(comp_dir)
            if expected_cifs and _is_valid_thermal_csv(
                comp_dir / "thermal_conductivity.csv",
                expected_cifs,
            ):
                valid.add(formula)
        return valid

    thermal_ready = False
    thermal_partial = False
    thermal_stage_status = "completed"
    thermal_failures = []
    thermal_oom_skips = []
    thermal_terminal_skips = []
    thermal_marked = bool(
        tracker
        and tracker.is_substep_completed(
            iteration_num,
            "structure_calculation",
            SUBSTEP_THERMAL,
        )
    )
    valid_thermal_materials = _valid_thermal_materials()
    thermal_partial = len(valid_thermal_materials) < len(materials)
    thermal_stage_status = (
        "completed_with_partial_results" if thermal_partial else "completed"
    )
    if thermal_marked and valid_thermal_materials:
        thermal_ready = True
        print("\n[SKIP] 子步骤 4.4（计算热导率）已有有效产物，跳过")
    else:
        if thermal_marked and tracker:
            print("\n[WARN] thermal 子步骤标记完成但没有有效产物，重置后重跑")
            tracker.reset_substep(
                iteration_num,
                "structure_calculation",
                SUBSTEP_THERMAL,
            )
        print(f"\n{'=' * 80}")
        print("4.4 Calculate thermal conductivity")
        print(f"{'=' * 80}")

        kappa_tasks = []
        for m in materials:
            formula = m["formula"]
            comp_dir = relax_output_dir / formula
            csv_file = comp_dir / "thermal_conductivity.csv"

            expected_cifs = _load_completed_phonon_relaxed_cifs(comp_dir)
            if comp_dir.exists() and expected_cifs:
                if not _is_valid_thermal_csv(csv_file, expected_cifs):
                    task_seed = _derive_seed(seed, "kappa", iteration_num, formula)
                    assigned_gpu = gpus[len(kappa_tasks) % len(gpus)]
                    kappa_tasks.append(
                        (str(comp_dir), formula, task_seed, assigned_gpu, expected_cifs)
                    )

        if kappa_tasks:
            print(f"[INFO] 正在为 {len(kappa_tasks)} 个材料计算热导率...")
            thermal_workers_per_gpu = 1

            def _run_kappa(task):
                diagnostic_path = (
                    Path(task[0]) / "worker_diagnostics" / "thermal.worker.log"
                )
                try:
                    # Keep the single-GPU path compatible with direct callers
                    # and tests. Multi-GPU work is isolated in spawn workers
                    # because kappa_lib keeps process-global model state.
                    if len(gpus) > 1:
                        result = run_kappa_task_with_timeout(task)
                    else:
                        result = calculate_kappa_worker(task)
                    if isinstance(result, dict) and not result.get("success"):
                        result.setdefault("worker_diagnostic", str(diagnostic_path))
                    return result
                except BaseException as exc:
                    import traceback

                    return {
                        "success": False,
                        "formula": task[1],
                        "error": str(exc),
                        "worker_error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                        "worker_diagnostic": str(diagnostic_path),
                    }

            kappa_results = run_tasks_by_gpu(
                kappa_tasks,
                gpus=gpus,
                workers_per_gpu=thermal_workers_per_gpu,
                task_device=lambda task: task[3],
                run_task=_run_kappa,
            )
            for idx, (task, res) in enumerate(zip(kappa_tasks, kappa_results)):
                if res["success"] and not _is_valid_thermal_csv(
                    Path(task[0]) / "thermal_conductivity.csv",
                    set(task[4]),
                ):
                    res = {
                        "success": False,
                        "formula": task[1],
                        "error": "Thermal conductivity CSV is missing or invalid",
                    }
                if res["success"]:
                    print(f"  [OK] {res['formula']} ({task[3]}): 计算完成")
                elif _is_oom_result(res):
                    thermal_oom_skips.append(res.get("formula"))
                    thermal_terminal_skips.append(
                        {
                            "formula": res.get("formula"),
                            "error": res.get("error") or "CUDA OOM",
                            "reason": "cuda_oom",
                            "worker_error_type": res.get("worker_error_type"),
                            "worker_exit_code": res.get("worker_exit_code"),
                            "traceback": res.get("traceback"),
                            "worker_diagnostic": res.get("worker_diagnostic"),
                        }
                    )
                    print(f"  [WARN] {res['formula']} ({task[3]}): CUDA OOM，跳过当前材料")
                else:
                    failure = {
                        "formula": res.get("formula"),
                        "error": res.get("error"),
                    }
                    thermal_failures.append(failure)
                    thermal_terminal_skips.append(
                        {
                            **failure,
                            "worker_error_type": res.get("worker_error_type"),
                            "worker_exit_code": res.get("worker_exit_code"),
                            "traceback": res.get("traceback"),
                            "worker_diagnostic": res.get("worker_diagnostic"),
                        }
                    )
                    print(f"  [WARN] {res['formula']}: 当前材料 thermal 失败，终止跳过 - {res.get('error')}")

                if (idx + 1) % 5 == 0:
                    safe_clear_memory(device)

            if thermal_failures:
                print(
                    f"[WARN] 热导率计算失败 {len(thermal_failures)} 个材料；"
                    "这些材料记录为 terminal skip，其余材料继续"
                )
            else:
                print("[OK] 热导率计算任务处理完成")
        else:
            print("[OK] 所有热导率均已计算，跳过")

        # A material-level thermal failure is terminal only when another
        # material leaves at least one valid thermal artifact for the round.
        # An all-failed/empty thermal chain is not a completed structure step.
        valid_thermal_materials = _valid_thermal_materials()
        thermal_ready = bool(valid_thermal_materials)
        thermal_partial = bool(thermal_terminal_skips) or len(valid_thermal_materials) < len(materials)
        if not relax_complete or not thermal_ready:
            thermal_stage_status = "incomplete"
        elif thermal_partial:
            thermal_stage_status = "completed_with_partial_results"
        else:
            thermal_stage_status = "completed"
        thermal_status_payload = {
            "iteration": iteration_num,
            "status": thermal_stage_status,
            "materials_total": len(materials),
            "materials_processed": len(kappa_tasks),
            "valid_materials": sorted(valid_thermal_materials),
            "valid_materials_count": len(valid_thermal_materials),
            "failed_materials": thermal_terminal_skips,
            "oom_materials_skipped": thermal_oom_skips,
            "partial": thermal_partial,
            "reason": None if thermal_ready else "no_valid_thermal_artifact",
        }
        temp_status_path = thermal_status_path.with_name(
            f".{thermal_status_path.name}.{os.getpid()}.tmp"
        )
        try:
            temp_status_path.write_text(
                json.dumps(thermal_status_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temp_status_path, thermal_status_path)
        finally:
            if temp_status_path.exists():
                try:
                    temp_status_path.unlink()
                except OSError:
                    pass

        if tracker and relax_complete and thermal_ready:
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_THERMAL,
                {
                    "status": thermal_stage_status,
                    "materials_total": len(materials),
                    "materials_processed": len(kappa_tasks),
                    "valid_materials_count": len(valid_thermal_materials),
                    "failed_materials": thermal_terminal_skips,
                    "oom_materials_skipped": thermal_oom_skips,
                    "partial": thermal_partial,
                },
            )
        elif tracker:
            tracker.update_substep(
                iteration_num,
                "structure_calculation",
                SUBSTEP_THERMAL,
                {
                    "status": thermal_stage_status,
                    "materials_total": len(materials),
                    "materials_processed": len(kappa_tasks),
                    "valid_materials_count": len(valid_thermal_materials),
                    "failed_materials": thermal_terminal_skips,
                    "oom_materials_skipped": thermal_oom_skips,
                    "partial": thermal_partial,
                },
                completed=False,
            )
        safe_clear_memory(device)

    thermal_complete = thermal_ready if not tracker else tracker.is_substep_completed(
        iteration_num, "structure_calculation", SUBSTEP_THERMAL
    )
    dedup_complete = dedup_success if not tracker else tracker.is_substep_completed(
        iteration_num, "structure_calculation", SUBSTEP_DEDUP
    )
    structure_complete = bool(relax_complete and thermal_complete and dedup_complete)
    structure_success = structure_complete
    generation_partial = generation_manifest.get("status") == "completed_with_partial_results"
    structure_partial = bool(generation_partial or relax_partial or thermal_partial)
    structure_status = (
        "completed_with_partial_results" if structure_partial else "completed"
    ) if structure_complete else "incomplete"
    structure_error = None
    if relax_complete and dedup_complete and not thermal_complete:
        structure_error = "No valid thermal conductivity artifact; iteration is incomplete"
        print(f"[ERROR] {structure_error}")

    return {
        "success": structure_success,
        "completed": structure_complete,
        "status": structure_status,
        "partial": structure_partial,
        "error": structure_error,
        "gen_output_dir": str(gen_output_dir),
        "relax_output_dir": str(relax_output_dir),
        "materials_generated": generated_formulas,
        "materials_skipped": materials_skipped,
        "skipped_details": skipped_details,
        "no_valid_structures": not bool(materials),
        "thermal_failures": thermal_failures if 'thermal_failures' in locals() else [],
        "thermal_oom_skips": thermal_oom_skips if 'thermal_oom_skips' in locals() else [],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()

    mock_materials = [{"formula": "AgBiS2"}]
    result = step_structure_calculation(args.iteration, mock_materials)
    print(f"\nResult: {result}")

