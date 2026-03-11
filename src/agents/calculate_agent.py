from __future__ import annotations

from agno.agent import Agent

from workflow.step_merge_results import step_merge_results
from workflow.step_structure_calculation import step_structure_calculation

calculate_agent = Agent(
    id="aslk-calculate-agent",
    name="ASLK Calculate Agent",
    role="Run structure generation and property calculations through registered tools.",
    instructions=[
        "Use step_structure_calculation for structure generation/relaxation/phonon/thermal tasks.",
        "Then use step_merge_results to merge phonon and thermal outputs.",
        "Return structured status and error details.",
    ],
    tools=[step_structure_calculation, step_merge_results],
)

__all__ = ["calculate_agent"]
