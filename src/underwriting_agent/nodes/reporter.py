"""Node 4 — emit a structured JSON audit report.

Trade-off: report is a dict on state (JSON-serializable) plus a frozen
AuditReport model. Persisting only the Pydantic object would hide
`model_dump(mode='json')` bugs until a downstream credit-ops system
chokes on a date. We serialize once at the edge.
"""

from __future__ import annotations

from underwriting_agent.schemas import (
    AgentState,
    AuditReport,
    PipelineStatus,
    append_trace,
    utcnow,
)


def reporter_node(state: AgentState) -> dict:
    statement = state.parsed_statement
    validation = state.validation
    assessment = state.risk_assessment

    if assessment and state.human_decision and state.human_decision.override_risk_score is not None:
        assessment = assessment.model_copy(
            update={"risk_score": state.human_decision.override_risk_score}
        )

    status = (
        PipelineStatus.NEED_MANUAL_INPUT
        if state.status is PipelineStatus.NEED_MANUAL_INPUT
        else PipelineStatus.COMPLETE
    )

    summary = _summary(state, assessment)
    report = AuditReport(
        generated_at=utcnow(),
        status=status,
        source_path=state.source_path,
        risk_assessment=assessment,
        validation_issues=validation.issues if validation else (),
        human_decision=state.human_decision,
        summary=summary,
        transaction_count=len(statement.transactions) if statement else 0,
    )
    return {
        "report": report.model_dump(mode="json"),
        "risk_assessment": assessment,
        "status": status,
        "trace": append_trace(state, f"report:{status.value}"),
    }


def _summary(state: AgentState, assessment) -> str:
    if state.status is PipelineStatus.NEED_MANUAL_INPUT:
        return f"Need Manual Input: {state.error_message or 'parser/validator could not certify the ledger.'}"
    if assessment is None:
        return "Need Manual Input: risk assessment missing."
    decision = state.human_decision
    human = ""
    if decision is not None:
        human = f" Human reviewer {'approved' if decision.approved else 'rejected'}."
    dti = f"{assessment.dti_ratio:.2%}" if assessment.dti_ratio is not None else "n/a"
    return f"Risk score {assessment.risk_score}/10; DTI {dti}.{human}"
