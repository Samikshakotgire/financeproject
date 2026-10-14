"""HITL breakpoint after Risk_Evaluator when score > threshold.

Trade-off: interrupt() (LangGraph durable execution) over a custom REST
callback. A webhook would couple us to an ops UI we do not have yet; the
graph checkpoint is enough for a reviewer to `Command(resume=...)` later.
We never auto-approve high scores — that would make the breakpoint dead code.
"""

from __future__ import annotations

from langgraph.types import interrupt

from underwriting_agent.schemas import AgentState, HumanDecision, PipelineStatus, append_trace


def human_review_node(state: AgentState) -> dict:
    assessment = state.risk_assessment
    payload = {
        "reason": "risk_score_exceeds_policy_threshold",
        "risk_score": assessment.risk_score if assessment else None,
        "dti_ratio": assessment.dti_ratio if assessment else None,
        "flags": [f.model_dump() for f in assessment.flags] if assessment else [],
        "instructions": (
            "Resume with {approved: bool, reviewer_notes: str, override_risk_score?: float}."
        ),
    }
    raw = interrupt(payload)
    decision = HumanDecision.model_validate(raw)
    return {
        "human_decision": decision,
        "status": PipelineStatus.RISK_SCORED,
        "trace": append_trace(
            state,
            f"human:{'approve' if decision.approved else 'reject'}",
        ),
    }
