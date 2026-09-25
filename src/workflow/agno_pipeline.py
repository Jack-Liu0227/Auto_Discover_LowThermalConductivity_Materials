from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from schemas.workflow_input import build_workflow_input_schema
from workflow.agno_steps import build_aslk_steps
from utils.workflow_resume import get_contiguous_completed_rounds


def build_workflow(
    config: dict[str, Any],
    tracker,
    start_iteration: int,
    max_iterations: int,
    initial_samples: list[dict[str, Any]] | None = None,
    workflow_id: str = "aslk-agno-workflow",
    workflow_name: str = "ASLK Agno Workflow",
    db_file: str | None = None,
):
    try:
        from agno.db.sqlite import SqliteDb
        from agno.workflow.workflow import Workflow
    except Exception as exc:
        raise RuntimeError("Agno is required. Install with `pip install agno`.") from exc

    db = None
    if db_file:
        db_path = Path(db_file)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = SqliteDb(db_file=str(db_path))

    steps = build_aslk_steps(
        config=config,
        tracker=tracker,
        start_iteration=start_iteration,
        max_iterations=max_iterations,
        initial_samples=initial_samples,
    )

    workflow_kwargs = {
        "id": workflow_id,
        "name": workflow_name,
        "description": "Train -> Bayes -> Screening -> Calculation -> Extract -> Theory Update",
        "steps": steps,
        "db": db,
    }
    try:
        sig = inspect.signature(Workflow.__init__)
        if "input_schema" in sig.parameters:
            # The schema is also used by the normal CLI workflow.  Use the
            # final effective config here so explicit CLI values are not
            # replaced by stale parameter-sheet defaults when Agno fills an
            # otherwise empty schema input.
            workflow_kwargs["input_schema"] = build_workflow_input_schema(config)
    except Exception:
        pass

    return Workflow(**workflow_kwargs)


def _build_workflow_run_input(workflow) -> Any:
    """Use schema-compatible input when Agno workflow input validation is enabled."""
    if getattr(workflow, "input_schema", None) is not None:
        return {}
    return "Start ASLK workflow"


def _resolve_results_root(config: dict[str, Any]) -> Path:
    path_config = config.get("path_config")
    if path_config is not None and getattr(path_config, "results_root", None) is not None:
        return Path(path_config.results_root)

    results_root = Path(config.get("results_root", "llm/results"))
    return results_root if results_root.is_absolute() else Path.cwd() / results_root


def run_workflow(
    config: dict[str, Any],
    tracker,
    start_iteration: int,
    max_iterations: int,
    initial_samples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    workflow = build_workflow(
        config=config,
        tracker=tracker,
        start_iteration=start_iteration,
        max_iterations=max_iterations,
        initial_samples=initial_samples,
        workflow_id="aslk-agno-workflow",
        workflow_name="ASLK Agno Workflow",
        db_file=str(_resolve_results_root(config) / "workflow.db"),
    )

    run_output = workflow.run(input=_build_workflow_run_input(workflow), stream=False)
    content = getattr(run_output, "content", {})
    if isinstance(content, dict):
        return content.get("all_results", [])
    return []


def serve_agentos(
    config: dict[str, Any],
    tracker,
    host: str = "127.0.0.1",
    port: int = 7777,
):
    try:
        from agno.os import AgentOS
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Agno AgentOS is required. Install with `pip install agno`.") from exc

    completed_rounds, last_completed = get_contiguous_completed_rounds(tracker)
    start_iteration = max(1, last_completed + 1)
    default_target = int(config.get("agentos_default_iterations", 3))
    effective_target = default_target

    workflow = build_workflow(
        config=config,
        tracker=tracker,
        start_iteration=start_iteration,
        max_iterations=effective_target,
        initial_samples=None,
        workflow_id="aslk-agentos-workflow",
        workflow_name="ASLK AgentOS Workflow",
        db_file=str(_resolve_results_root(config) / "agentos.db"),
    )

    # Give AgentOS an explicit DB as recommended when tracing is enabled.
    try:
        from agno.db.sqlite import SqliteDb

        os_db_path = _resolve_results_root(config) / "agentos.db"
        os_db_path.parent.mkdir(parents=True, exist_ok=True)
        os_db = SqliteDb(db_file=str(os_db_path))
    except Exception:
        os_db = None

    agent_os = AgentOS(
        id="aslk-agentos",
        description="ASLK Agno AgentOS runtime",
        workflows=[workflow],
        tracing=True,
        db=os_db,
    )

    app = agent_os.get_app()
    ws_ping_interval = config.get("agentos_ws_ping_interval", None)
    ws_ping_timeout = config.get("agentos_ws_ping_timeout", None)
    print(
        f"[agentos] completed_rounds={completed_rounds}, start_iteration={start_iteration}, "
        f"effective_target={effective_target}"
    )
    print(
        f"[agentos] ws_ping_interval={ws_ping_interval}, ws_ping_timeout={ws_ping_timeout} "
        "(set to None to disable server ping)"
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        ws_ping_interval=ws_ping_interval,
        ws_ping_timeout=ws_ping_timeout,
    )


__all__ = ["build_workflow", "run_workflow", "serve_agentos"]
