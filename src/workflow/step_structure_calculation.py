# -*- coding: utf-8 -*-
"""
Workflow Step 4: Structure Generation and Calculation
负责结构生成、弛豫、声子计算（记录到 CSV）、结构去重（避免重复计算）、
以及热导率计算。声子计算与弛豫阶段合并执行，不再单独补算。
"""
import os
import sys
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# 所有 pymatgen 相关导入都延迟到函数内部
# 这是为了避免 Windows 上 torch/pymatgen 导入顺序冲突

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.structure_parallel import generate_structures_parallel
from utils.types import Composition, CrystalStructure


def safe_clear_memory(device="cuda"):
    """
    安全清理内存，并捕获可能的 OOM 错误。

    Args:
        device: 设备类型
    """
    import gc
    gc.collect()

    if device == "cuda":
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


def relax_structure_worker(args):
    """Relax one structure and run phonon calculation."""
    cif_path, formula, relax_base_dir, pressure, gpu_device = args
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    if gpu_device and ":" in gpu_device:
        gpu_id = gpu_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"  [Worker {os.getpid()}] 璁剧疆GPU: {gpu_device} (CUDA_VISIBLE_DEVICES={gpu_id})")

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
        print(f"  [Worker] GPU娓呯悊璀﹀憡: {e}")

    try:
        from tools.mattersim_wrapper import MattersimWrapper
        from ase.io import write as ase_write
        from io import StringIO
        from pathlib import Path
        from pymatgen.core import Composition as PMGComposition, Structure
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
            space_group="P1",
        )

        mattersim = MattersimWrapper()
        print(f"  姝ｅ湪寮涜鲍: {cif_path.name}")
        response = mattersim.relax_structure(crystal_struct, pressure=pressure)

        if response.is_success():
            relaxed_atoms = response.result
            ase_write(str(output_file), relaxed_atoms, format="cif")
            print(f"  寮涜鲍鎴愬姛: {output_file}")

            phonon_success = False
            phonon_error = None
            has_imaginary = "鏈煡"
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
                    space_group="P1",
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
                phonon_error = str(phonon_exc)

            return {
                "success": True,
                "formula": formula,
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

        print(f"  寮涜鲍澶辫触: {cif_path.name} - {response.error}")
        return {
            "success": False,
            "formula": formula,
            "cif_file": cif_path.name,
            "error": response.error,
        }

    except Exception as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            print(f"  鈿狅笍 寮涜鲍OOM: {cif_path.name} - GPU鍐呭瓨涓嶈冻锛岃烦杩囨缁撴瀯")
            return {
                "success": False,
                "formula": formula,
                "cif_file": Path(cif_path).name,
                "error": "CUDA OOM",
                "skipped": True,
            }
        print(f"  寮涜鲍寮傚父: {e}")
        return {
            "success": False,
            "formula": formula,
            "cif_file": Path(cif_path).name,
            "error": error_msg,
        }
    finally:
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


def _timeout_worker(task, queue):
    res = relax_structure_worker(task)
    queue.put(res)


def run_relax_task_with_timeout(task, timeout_sec: int):
    import multiprocessing
    from pathlib import Path as _Path

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    proc = ctx.Process(target=_timeout_worker, args=(task, result_queue))
    proc.start()
    proc.join(timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "success": False,
            "formula": task[1],
            "cif_file": _Path(task[0]).name,
            "error": "Timeout",
        }

    try:
        return result_queue.get_nowait()
    except Exception:
        return {
            "success": False,
            "formula": task[1],
            "cif_file": _Path(task[0]).name,
            "error": "No result",
        }


def calculate_kappa_worker(args):
    """鍗曚釜鏉愭枡鐨勭儹瀵肩巼璁＄畻浠诲姟"""
    comp_dir, formula = args

    import time
    time.sleep(0.5)

    try:
        from pathlib import Path
        from pymatgen.core import Composition as PMGComposition, Structure
        from pymatgen.io.vasp import Poscar as PmgPoscar

        comp_dir = Path(comp_dir)
        cif_files = sorted(comp_dir.glob("*.cif"))
        if not cif_files:
            return {"success": False, "formula": formula, "error": "No CIF files found"}

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
                    space_group="P1",
                )
                structures.append(cs)
            except Exception as e:
                print(f"  读取 CIF 失败 {cif_file}: {e}")

        if not structures:
            return {"success": False, "formula": formula, "error": "No valid structures loaded"}

        from tools.crystallm_wrapper import CrystaLLMWrapper as _CrystaLLMWrapper
        wrapper = _CrystaLLMWrapper(output_dir=str(comp_dir))
        print(f"  正在计算热导率 {formula}（{len(structures)} 个结构）...")
        wrapper._calculate_and_save_thermal_conductivity(comp_dir, structures, composition)

        return {"success": True, "formula": formula}

    except Exception as e:
        print(f"  热导率计算异常 {formula} - {e}")
        return {"success": False, "formula": formula, "error": str(e)}


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
    tracker=None,
    allow_partial_completion: bool = False,
    path_config=None,
    relax_timeout_sec: int = 120,
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
        pressure: 弛豫压力
        device: 计算设备
        gpus: GPU 列表；如果为 None 则使用 device 参数
        results_root: 结果存储根目录
        tracker: 进度跟踪器实例

    Returns:
        dict: 包含计算状态的信息
    """
    print("=" * 80)
    print(f"步骤 4: 结构生成与计算 (Iteration {iteration_num})")
    print("=" * 80)

    SUBSTEP_GENERATION = "generation"
    SUBSTEP_RELAXATION = "relaxation"
    SUBSTEP_THERMAL = "thermal_conductivity"
    SUBSTEP_DEDUP = "deduplication"
    SUBSTEP_PHONON = "phonon_spectrum"
    relax_complete = False

    gen_output_dir = project_root / results_root / f"iteration_{iteration_num}" / "processed_structures"
    relax_output_dir = project_root / results_root / f"iteration_{iteration_num}" / "MyRelaxStructure"

    def _append_relax_phonon_log(comp_dir: Path, row: dict):
        """鍐欏叆寮涜鲍/澹板瓙璋盋SV璁板綍"""
        log_file = comp_dir / "relax_phonon_results.csv"
        fieldnames = [
            "Formula",
            "CIF_File",
            "Relaxed_CIF",
            "Relax_Success",
            "Relax_Error",
            "Phonon_Success",
            "Phonon_Error",
            "Has_Imaginary_Freq",
            "Min_Frequency",
            "Gamma_Min_Optical",
            "Gamma_Max_Acoustic",
        ]
        comp_dir.mkdir(parents=True, exist_ok=True)
        write_header = not log_file.exists()
        with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in fieldnames})

    def _get_relaxed_cifs(formula: str):
        comp_dir = relax_output_dir / formula
        if comp_dir.exists():
            return sorted(comp_dir.glob("*.cif"))
        return []

    def _count_generated_cifs(formula: str):
        comp_dir = gen_output_dir / formula
        processed_dir = comp_dir / "processed"
        if processed_dir.exists():
            return len(list(processed_dir.glob("*.cif")))
        if comp_dir.exists():
            return len(list(comp_dir.glob("*.cif")))
        return 0

    def _load_logged_cifs(comp_dir: Path):
        log_file = comp_dir / "relax_phonon_results.csv"
        if not log_file.exists():
            return set()
        try:
            with open(log_file, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                return {row.get("CIF_File") for row in reader if row.get("CIF_File")}
        except Exception:
            return set()

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
    generation_marked = tracker and tracker.is_substep_completed(
        iteration_num, "structure_calculation", SUBSTEP_GENERATION
    )
    if generation_marked:
        missing = [m["formula"] for m in materials if _count_generated_cifs(m["formula"]) < n_structures]
        if missing:
            print(f"[WARN] 结构生成进度记录已完成，但缺少结果: {missing}")
            if tracker:
                tracker.reset_substep(iteration_num, "structure_calculation", SUBSTEP_GENERATION)
            generation_marked = False

    if generation_marked:
        print("\n[SKIP] 子步骤 4.1（生成结构）已完成，跳过")
    else:
        print(f"\n{'=' * 80}")
        print("4.1 生成晶体结构")
        print(f"{'=' * 80}")

        materials_to_gen = []
        materials_existing = []
        for m in materials:
            formula = m["formula"]
            comp_dir = gen_output_dir / formula
            processed_dir = comp_dir / "processed"

            existing_cifs = []
            if processed_dir.exists():
                existing_cifs = sorted(processed_dir.glob("*.cif"))
            if not existing_cifs and comp_dir.exists():
                existing_cifs = sorted(comp_dir.glob("*.cif"))

            if len(existing_cifs) >= n_structures:
                print(f"  [SKIP] {formula}: 已存在 {len(existing_cifs)} 个结构，跳过生成")
                materials_existing.append(formula)
            else:
                materials_to_gen.append(m)

        if tracker:
            tracker.update_substep(
                iteration_num,
                "structure_calculation",
                SUBSTEP_GENERATION,
                {
                    "materials_total": len(materials),
                    "materials_existing": materials_existing,
                    "materials_pending": [m["formula"] for m in materials_to_gen],
                },
            )

        generated_formulas = []
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
            )

            success_count = sum(1 for r in gen_results if r.get("success"))
            generated_formulas = [r.get("formula") for r in gen_results if r.get("success")]
            print(f"[OK] 结构生成完成！成功: {success_count}/{len(materials_to_gen)}")

            if success_count < len(materials_to_gen):
                print(f"[WARN] generation failed for {len(materials_to_gen) - success_count} materials")
        else:
            print("[OK] 所有结构已生成，检测到后跳过")

        if tracker:
            tracker.update_substep(
                iteration_num,
                "structure_calculation",
                SUBSTEP_GENERATION,
                {
                    "materials_generated": generated_formulas,
                    "materials_existing": materials_existing,
                },
            )
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_GENERATION,
                {"materials_processed": len(materials)},
            )
        safe_clear_memory(device)

    # === 4.2 弛豫 + 声子计算 ===
    if tracker and tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_RELAXATION):
        print("\n[SKIP] 子步骤 4.2（弛豫 + 声子计算）已完成，跳过")
        relax_complete = True
    else:
        print(f"\n{'=' * 80}")
        print("4.2 Relax structures + phonon calculation")
        print(f"{'=' * 80}")

        if not gpus:
            gpus = [device]

        if len(gpus) == 1 and gpus[0].startswith("cuda:"):
            gpu_id = gpus[0].split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(f"  [INFO] 单 GPU 模式: 限制 CUDA_VISIBLE_DEVICES={gpu_id}")

        relax_tasks = []
        task_idx = 0
        materials_with_structures = []
        source_cifs_map = {}
        for m in materials:
            formula = m["formula"]
            gen_comp_dir = gen_output_dir / formula
            relax_comp_dir = relax_output_dir / formula
            processed_dir = gen_comp_dir / "processed"
            logged_cifs = _load_logged_cifs(relax_comp_dir)

            search_dir = processed_dir if processed_dir.exists() else gen_comp_dir

            if search_dir.exists():
                existing_cifs = sorted(search_dir.glob("*.cif"))
                source_cifs_map[formula] = {c.name for c in existing_cifs}
                if len(existing_cifs) >= n_structures:
                    materials_with_structures.append(formula)
                for cif_file in existing_cifs:
                    relaxed_cif = relax_comp_dir / cif_file.name
                    if cif_file.name in logged_cifs:
                        continue
                    if not relaxed_cif.exists():
                        assigned_gpu = gpus[task_idx % len(gpus)]
                        relax_tasks.append((str(cif_file), formula, str(relax_output_dir), pressure, assigned_gpu))
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

        if relax_tasks:
            if gpus and len(gpus) > 1:
                requested_workers = len(gpus) * relax_workers
                actual_workers = len(gpus)
                print(f"Total structures to relax: {len(relax_tasks)}")
                print(f"  - GPU鏁伴噺: {len(gpus)}")
                print(f"  - 每 GPU 并行数: {relax_workers}（已限制为 1）")
                print(f"  - 总并行数: {actual_workers}（原本 {requested_workers}）")
                print("  - GPU 任务分配:")
                gpu_task_count = {gpu: 0 for gpu in gpus}
                for task in relax_tasks:
                    gpu_task_count[task[4]] += 1
                for gpu, count in gpu_task_count.items():
                    print(f"      {gpu}: {count} tasks")
            else:
                actual_workers = 1
                print(f"共有 {len(relax_tasks)} 个结构需要弛豫，使用 1 个并行进程...")

            relaxed_count = 0
            phonon_success_count = 0
            processed_count = 0

            materials_progress = {}
            if tracker:
                current_meta = tracker.get_substep_metadata(
                    iteration_num, "structure_calculation", SUBSTEP_RELAXATION
                ) or {}
                materials_progress = current_meta.get("materials_progress", {})

            for i, task in enumerate(relax_tasks):
                res = run_relax_task_with_timeout(task, relax_timeout_sec)
                processed_count += 1

                formula = res.get("formula")
                if formula in materials_progress:
                    materials_progress[formula]["processed"] += 1

                relax_success = bool(res.get("success"))
                if relax_success:
                    relaxed_count += 1
                    if formula in materials_progress:
                        materials_progress[formula]["relax_success"] += 1

                phonon_success = bool(res.get("phonon_success"))
                if phonon_success:
                    phonon_success_count += 1
                    if formula in materials_progress:
                        materials_progress[formula]["phonon_success"] += 1

                log_row = {
                    "Formula": formula,
                    "CIF_File": res.get("cif_file"),
                    "Relaxed_CIF": res.get("file"),
                    "Relax_Success": "Y" if relax_success else "N",
                    "Relax_Error": res.get("relax_error") or res.get("error"),
                    "Phonon_Success": "Y" if phonon_success else "N",
                    "Phonon_Error": res.get("phonon_error"),
                    "Has_Imaginary_Freq": res.get("has_imaginary"),
                    "Min_Frequency": res.get("min_frequency"),
                    "Gamma_Min_Optical": res.get("gamma_min_optical"),
                    "Gamma_Max_Acoustic": res.get("gamma_max_acoustic"),
                }
                _append_relax_phonon_log(relax_output_dir / formula, log_row)

                print(f"  [{i + 1}/{len(relax_tasks)}] 寮涜鲍瀹屾垚")

                if tracker:
                    tracker.update_substep(
                        iteration_num,
                        "structure_calculation",
                        SUBSTEP_RELAXATION,
                        {
                            "structures_processed": processed_count,
                            "relax_success": relaxed_count,
                            "phonon_success": phonon_success_count,
                            "last_formula": formula,
                            "materials_progress": materials_progress,
                        },
                    )

                import time
                safe_clear_memory(device)
                if res.get("error") in {"CUDA OOM", "Timeout"}:
                    print("  [WARN] 检测到 OOM 或超时，延长等待并加强清理...")
                    for _ in range(3):
                        safe_clear_memory(device)
                    time.sleep(8)
                else:
                    time.sleep(3)

                if (i + 1) % 10 == 0:
                    print(f"  [Memory] task {i + 1} completed, running deep cleanup...")
                    for _ in range(3):
                        safe_clear_memory(device)
                    time.sleep(2)

            print(f"弛豫完成！成功: {relaxed_count}/{len(relax_tasks)}")
        else:
            print("[OK] All structures already relaxed, skipping")

        # 鍒ゆ柇寮涜鲍鏄惁鐪熸瀹屾垚锛氭瘡涓簮缁撴瀯閮芥湁璁板綍锛堟垚鍔熸垨澶辫触锛夋垨宸叉湁寮涜鲍鏂囦欢
        relax_complete = True
        for m in materials:
            formula = m["formula"]
            src_cifs = source_cifs_map.get(formula, set())
            if not src_cifs:
                relax_complete = False
                break
            comp_dir = relax_output_dir / formula
            logged = _load_logged_cifs(comp_dir)
            relaxed = {p.name for p in comp_dir.glob("*.cif")} if comp_dir.exists() else set()
            done = logged | relaxed
            if not src_cifs.issubset(done):
                relax_complete = False
                break

        if tracker and relax_complete:
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_RELAXATION,
                {
                    "structures_relaxed": len(relax_tasks),
                    "relax_success": relaxed_count if relax_tasks else 0,
                    "phonon_success": phonon_success_count if relax_tasks else 0,
                },
            )
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_PHONON,
                {"combined_with_relaxation": True},
            )
        safe_clear_memory(device)

    # === 4.3 弛豫结构去重（避免重复计算） ===
    if tracker and tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_DEDUP):
        print("\n[SKIP] 子步骤 4.3（结构去重）已完成，跳过")
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

        except Exception as e:
            print(f"鈿狅笍 缁撴瀯鍘婚噸澶辫触: {e}")
            print("缁х画鎵ц鍚庣画璁＄畻...")

        if tracker and relax_complete:
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_DEDUP,
                {"materials_deduplicated": len(materials)},
            )
        safe_clear_memory(device)

    # === 4.4 计算热导率 ===
    if tracker and tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_THERMAL):
        print("\n[SKIP] 子步骤 4.4（计算热导率）已完成，跳过")
    else:
        print(f"\n{'=' * 80}")
        print("4.4 Calculate thermal conductivity")
        print(f"{'=' * 80}")

        kappa_tasks = []
        for m in materials:
            formula = m["formula"]
            comp_dir = relax_output_dir / formula
            csv_file = comp_dir / "thermal_conductivity.csv"

            if comp_dir.exists() and _get_relaxed_cifs(formula):
                if not csv_file.exists():
                    kappa_tasks.append((str(comp_dir), formula))

        if kappa_tasks:
            print(f"[INFO] 正在为 {len(kappa_tasks)} 个材料计算热导率...")
            for idx, task in enumerate(kappa_tasks):
                res = calculate_kappa_worker(task)
                if res["success"]:
                    print(f"  [OK] {res['formula']}: 计算完成")
                else:
                    print(f"  [FAIL] {res['formula']}: 失败 - {res.get('error')}")

                if (idx + 1) % 5 == 0:
                    safe_clear_memory(device)

            print("[OK] 热导率计算完成")
        else:
            print("[OK] 所有热导率均已计算，跳过")


        if tracker and relax_complete:
            tracker.mark_substep_completed(
                iteration_num,
                "structure_calculation",
                SUBSTEP_THERMAL,
                {"materials_processed": len(kappa_tasks)},
            )
        safe_clear_memory(device)

    return {
        "success": True,
        "completed": bool(relax_complete)
        and (not tracker or tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_DEDUP))
        and (not tracker or tracker.is_substep_completed(iteration_num, "structure_calculation", SUBSTEP_THERMAL)),
        "gen_output_dir": str(gen_output_dir),
        "relax_output_dir": str(relax_output_dir),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()

    mock_materials = [{"formula": "AgBiS2"}]
    result = step_structure_calculation(args.iteration, mock_materials)
    print(f"\n缁撴灉: {result}")

