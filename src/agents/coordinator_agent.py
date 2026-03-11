from __future__ import annotations

from agno.agent import Agent

coordinator_agent = Agent(
    id="aslk-coordinator-agent",
    name="ASLK Coordinator Agent",
    role="Coordinate workflow state transitions and step sequencing.",
    instructions=[
        "Follow workflow order strictly: train -> bayes -> screening -> calculate -> extract -> theory update.",
        "Track failures and stop iteration immediately when a hard failure occurs.",
        "Expose machine-readable status for each step.",
    ],
)

__all__ = ["coordinator_agent"]
