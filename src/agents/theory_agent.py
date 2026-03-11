from __future__ import annotations

from agno.agent import Agent

from workflow.step_update_data_doc import step_update_data_and_doc

theory_agent = Agent(
    id="aslk-theory-agent",
    name="ASLK Theory Agent",
    role="Update theory document and dataset refresh decisions from extraction outputs.",
    instructions=[
        "When success materials exist, update the theory document.",
        "When no success materials exist, keep/copy previous theory document and report reason.",
        "Return updated document path, data path, and strategy hints for next iteration.",
    ],
    tools=[step_update_data_and_doc],
)

__all__ = ["theory_agent"]
