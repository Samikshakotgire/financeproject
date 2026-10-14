from pathlib import Path

from underwriting_agent.graph import compile_underwriting_graph, route_after_risk
from underwriting_agent.schemas import AgentState, PipelineStatus, RiskAssessment

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_statement.csv"
HIGH_RISK = ROOT / "examples" / "high_risk_statement.csv"


def test_happy_path_csv_completes_without_human():
    graph = compile_underwriting_graph()
    result = graph.invoke(
        AgentState(source_path=str(SAMPLE)),
        {"configurable": {"thread_id": "happy"}},
    )
    state = AgentState.model_validate(result)
    assert state.status is PipelineStatus.COMPLETE
    assert state.report is not None
    assert state.risk_assessment is not None
    assert state.risk_assessment.risk_score <= 7


def test_missing_file_need_manual_input():
    graph = compile_underwriting_graph()
    result = graph.invoke(
        AgentState(source_path="C:/definitely-missing-statement.csv"),
        {"configurable": {"thread_id": "missing"}},
    )
    state = AgentState.model_validate(result)
    assert state.status is PipelineStatus.NEED_MANUAL_INPUT
    assert state.report["status"] == PipelineStatus.NEED_MANUAL_INPUT.value
    assert "parse_failed" in (state.error_message or "")


def test_high_risk_routes_to_human_review():
    assessment = RiskAssessment(
        risk_score=8.2,
        rationale="test",
        used_llm=False,
    )
    state = AgentState(
        source_path="x.csv",
        status=PipelineStatus.AWAITING_HUMAN,
        risk_assessment=assessment,
    )
    assert route_after_risk(state) == "human_review"


def test_high_risk_invoke_interrupts():
    graph = compile_underwriting_graph()
    result = graph.invoke(
        AgentState(source_path=str(HIGH_RISK)),
        {"configurable": {"thread_id": "hitl"}},
    )
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if interrupts is None:
        # LangGraph may return a state object with interrupts on the checkpoint.
        snapshot = graph.get_state({"configurable": {"thread_id": "hitl"}})
        assert snapshot.next == ("human_review",) or snapshot.tasks
    else:
        assert interrupts
