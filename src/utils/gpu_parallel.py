"""Utilities for explicit, spawn-safe GPU task scheduling.

The workflow runs independent material calculations rather than one distributed
model. Each GPU lane is therefore isolated with a bounded executor, while
GPU-bound work can optionally run in a fresh ``spawn`` process for timeout and
CUDA-context isolation.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Callable, Sequence, TypeVar
import json
import multiprocessing
import os
import sys
import traceback


TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


class SpawnTaskStartError(RuntimeError):
    """The worker process could not be started."""


class SpawnTaskError(RuntimeError):
    """The worker process started but returned an error or no result."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: int | None = None,
        diagnostic_path: str | Path | None = None,
        traceback_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.diagnostic_path = str(diagnostic_path) if diagnostic_path else None
        self.traceback_text = traceback_text


def normalize_gpu_list(
    gpus: Sequence[str] | None,
    *,
    device: str = "cuda",
) -> list[str]:
    """Normalize and validate the shape of the configured device list.

    Device indices are PyTorch logical CUDA indices, i.e. the indices visible
    under the current ``CUDA_VISIBLE_DEVICES`` setting. The function does not
    import torch; runtime availability is checked separately so unit tests and
    CPU-only callers can still construct a workflow configuration.
    """
    requested_device = str(device or "cuda").strip().lower()
    if requested_device == "cpu":
        return ["cpu"]

    raw_devices = list(gpus) if gpus else [requested_device]
    normalized = [str(item or "").strip().lower() for item in raw_devices]
    if not normalized or any(not item for item in normalized):
        raise ValueError("GPU configuration must contain at least one non-empty device")

    for item in normalized:
        if item == "cpu":
            continue
        if item == "cuda":
            continue
        if not item.startswith("cuda:"):
            raise ValueError(
                f"Unsupported GPU device {item!r}; use 'cuda', 'cuda:N', or 'cpu'"
            )
        index_text = item.split(":", 1)[1]
        if not index_text.isdigit():
            raise ValueError(f"Invalid CUDA device index: {item!r}")

    if "cpu" in normalized and any(item != "cpu" for item in normalized):
        raise ValueError("CPU and CUDA devices cannot be mixed in one task pool")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate GPU devices are not allowed: {normalized!r}")
    return normalized


def validate_gpu_devices(gpus: Sequence[str]) -> list[str]:
    """Validate that every configured CUDA device is visible to PyTorch."""
    normalized = normalize_gpu_list(gpus)
    if normalized == ["cpu"]:
        return normalized

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA devices were configured but PyTorch is not installed"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA devices were configured ({normalized!r}), but CUDA is unavailable"
        )

    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        raise RuntimeError(
            f"CUDA devices were configured ({normalized!r}), but PyTorch sees none"
        )

    for device_name in normalized:
        if device_name == "cuda":
            continue
        index = int(device_name.split(":", 1)[1])
        if index >= device_count:
            raise RuntimeError(
                f"Configured device {device_name} is unavailable: "
                f"PyTorch sees {device_count} CUDA device(s)"
            )
    return normalized


def set_process_cuda_device(device: str) -> None:
    """Select the explicit CUDA device inside an isolated worker process."""
    normalized = str(device or "").strip().lower()
    if normalized == "cpu":
        return
    if normalized != "cuda" and not normalized.startswith("cuda:"):
        raise ValueError(f"Unsupported worker device: {device!r}")

    validate_gpu_devices([normalized])
    import torch

    torch.cuda.set_device(0 if normalized == "cuda" else normalized)


def run_tasks_by_gpu(
    tasks: Sequence[TaskT],
    *,
    gpus: Sequence[str],
    workers_per_gpu: int,
    task_device: Callable[[TaskT], str],
    run_task: Callable[[TaskT], ResultT],
    on_result: Callable[[int, TaskT, ResultT], None] | None = None,
) -> list[ResultT]:
    """Run tasks concurrently in per-GPU lanes and preserve input ordering.

    A separate executor is created for every configured device. This is
    important: a single global executor can start a second task on GPU 0 as
    soon as a GPU 1 task completes, even while GPU 0 is still busy. Per-GPU
    executors enforce the requested bound independently for each device.

    ``run_task`` is normally a small parent-side wrapper around a spawn child;
    the threads here only coordinate child processes and never mutate CUDA
    process-global state.
    """
    if not tasks:
        return []
    normalized_gpus = normalize_gpu_list(gpus)
    lane_width = max(1, int(workers_per_gpu))
    lanes: dict[str, list[tuple[int, TaskT]]] = {gpu: [] for gpu in normalized_gpus}

    for index, task in enumerate(tasks):
        assigned = str(task_device(task) or "").strip().lower()
        if assigned not in lanes:
            raise ValueError(
                f"Task {index} is assigned to {assigned!r}, "
                f"which is not in configured GPUs {normalized_gpus!r}"
            )
        lanes[assigned].append((index, task))

    results: list[ResultT | None] = [None] * len(tasks)
    executors: dict[str, ThreadPoolExecutor] = {}
    future_to_index: dict[Future[ResultT], int] = {}
    try:
        for gpu in normalized_gpus:
            lane_tasks = lanes[gpu]
            if not lane_tasks:
                continue
            executor = ThreadPoolExecutor(
                max_workers=lane_width,
                thread_name_prefix=f"gpu-{gpu.replace(':', '-')}",
            )
            executors[gpu] = executor
            for index, task in lane_tasks:
                future_to_index[executor.submit(run_task, task)] = index

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            result = future.result()
            results[index] = result
            if on_result is not None:
                # The callback runs in this parent-side collector thread, not in
                # a GPU worker. This keeps shared CSV/status writes single-writer
                # while still allowing durable per-task checkpoints.
                on_result(index, tasks[index], result)
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)

    missing = [index for index, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"GPU task scheduler returned no result for task(s): {missing}")
    return results  # type: ignore[return-value]


def _append_spawn_diagnostic(path: str | Path | None, payload: dict) -> None:
    """Append a JSONL diagnostic without allowing logging to break a worker."""
    if not path:
        return
    try:
        diagnostic_path = Path(path)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with diagnostic_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
    except Exception:
        # Diagnostics are best effort and must never mask the original failure.
        pass


def _spawn_worker_entry(
    worker: Callable[[TaskT], ResultT],
    task: TaskT,
    result_queue,
    diagnostic_path: str | Path | None = None,
) -> None:
    """Execute a top-level worker and serialize either its result or error.

    The child also redirects Python and OS-level stdout/stderr to a per-task
    file. This preserves useful native-library messages when a process exits
    before it can put a result on the multiprocessing queue.
    """
    output_handle = None
    saved_stdout_fd = None
    saved_stderr_fd = None
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    try:
        if diagnostic_path:
            diagnostic_file = Path(diagnostic_path)
            diagnostic_file.parent.mkdir(parents=True, exist_ok=True)
            output_handle = diagnostic_file.open("a", encoding="utf-8", buffering=1)
            saved_stdout_fd = os.dup(1)
            saved_stderr_fd = os.dup(2)
            os.dup2(output_handle.fileno(), 1)
            os.dup2(output_handle.fileno(), 2)
            sys.stdout = output_handle
            sys.stderr = output_handle
            print(json.dumps({"event": "worker_started", "pid": os.getpid()}), flush=True)

        try:
            result_queue.put(("ok", worker(task)))
        except BaseException as exc:  # child must report even non-standard failures
            error_traceback = traceback.format_exc()
            _append_spawn_diagnostic(
                diagnostic_path,
                {
                    "event": "python_exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": error_traceback,
                },
            )
            try:
                result_queue.put(("error", type(exc).__name__, str(exc), error_traceback))
            except BaseException:
                # A broken queue must not hide the diagnostic file.
                pass
    finally:
        try:
            if output_handle is not None:
                output_handle.flush()
        except Exception:
            pass
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
                os.close(saved_stdout_fd)
            except Exception:
                pass
        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
            except Exception:
                pass
        if output_handle is not None:
            try:
                output_handle.close()
            except Exception:
                pass


def run_spawn_task(
    worker: Callable[[TaskT], ResultT],
    task: TaskT,
    *,
    timeout_sec: float | None = None,
    diagnostic_path: str | Path | None = None,
) -> ResultT:
    """Run a top-level worker in a fresh spawn process.

    ``timeout_sec=None`` waits until completion. A timed-out process is
    forcefully terminated so CUDA contexts and native resources do not remain
    attached to the parent workflow.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_spawn_worker_entry,
        args=(worker, task, result_queue, diagnostic_path),
    )
    _append_spawn_diagnostic(
        diagnostic_path,
        {"event": "parent_created", "timeout_sec": timeout_sec},
    )
    try:
        try:
            process.start()
        except Exception as exc:
            _append_spawn_diagnostic(
                diagnostic_path,
                {"event": "process_start_error", "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise SpawnTaskStartError(str(exc)) from exc
        _append_spawn_diagnostic(
            diagnostic_path,
            {"event": "parent_started", "pid": process.pid},
        )
        process.join(timeout_sec)
        if process.is_alive():
            if hasattr(process, "kill"):
                process.kill()
            else:
                process.terminate()
            process.join(5)
            _append_spawn_diagnostic(
                diagnostic_path,
                {"event": "timeout", "exit_code": process.exitcode, "timeout_sec": timeout_sec},
            )
            raise TimeoutError(f"Worker timed out after {timeout_sec}s")

        try:
            message = result_queue.get(timeout=2.0)
        except Empty:
            exit_code = process.exitcode
            _append_spawn_diagnostic(
                diagnostic_path,
                {
                    "event": "worker_no_result",
                    "exit_code": exit_code,
                    "message": "Worker exited without a queue result",
                },
            )
            raise SpawnTaskError(
                "Worker exited without a result"
                if exit_code in (0, None)
                else f"Worker exited with code {exit_code}",
                exit_code=exit_code,
                diagnostic_path=diagnostic_path,
            )

        if message[0] == "ok":
            return message[1]
        _, error_type, error_message, error_traceback = message
        _append_spawn_diagnostic(
            diagnostic_path,
            {
                "event": "worker_reported_error",
                "error_type": error_type,
                "error": error_message,
                "traceback": error_traceback,
            },
        )
        raise SpawnTaskError(
            f"{error_type}: {error_message}\n{error_traceback}",
            exit_code=process.exitcode,
            diagnostic_path=diagnostic_path,
            traceback_text=error_traceback,
        )
    finally:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
        try:
            process.close()
        except Exception:
            pass
