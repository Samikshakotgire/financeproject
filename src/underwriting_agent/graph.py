"""Compile the underwriting state machine.

Linear happy path: parser -> validator -> risk_evaluator -> reporter.
Conditional edges encode two policy choices:
  1. Fail closed to reporter (NEED_MANUAL_INPUT) on parse/validate errors.
  2. Park on human_review when risk_score > 7, then resume into reporter.

Checkpointer is required for interrupt(); MemorySaver is enough for local
dev. Swap for PostgresSaver in production so a reviewer crash does not
lose the statement bytes already parsed.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from underwriting_agent.config import get_settings
from underwriting_agent.nodes.human_review import human_review_node
from underwriting_agent.nodes.parser import parser_node
from underwriting_agent.nodes.reporter import reporter_node
from underwriting_agent.nodes.risk_evaluator import risk_evaluator_node
from underwriting_agent.nodes.validator import validator_node
from underwriting_agent.schemas import AgentState, PipelineStatus


def route_after_parser(state: AgentState) -> str:
    if state.status is PipelineStatus.NEED_MANUAL_INPUT:
        return "reporter"
    return "validator"


def route_after_validator(state: AgentState) -> str:
    if state.status is PipelineStatus.NEED_MANUAL_INPUT:
        return "reporter"
    return "risk_evaluator"


def route_after_risk(state: AgentState) -> str:
    threshold = get_settings().human_review_risk_threshold
    score = state.risk_assessment.risk_score if state.risk_assessment else 0.0
    if state.status is PipelineStatus.AWAITING_HUMAN or score > threshold:
        return "human_review"
    return "reporter"


def build_underwriting_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("parser", parser_node)
    graph.add_node("validator", validator_node)
    graph.add_node("risk_evaluator", risk_evaluator_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("reporter", reporter_node)

    graph.add_edge(START, "parser")
    graph.add_conditional_edges(
        "parser",
        route_after_parser,
        {"validator": "validator", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {"risk_evaluator": "risk_evaluator", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "risk_evaluator",
        route_after_risk,
        {"human_review": "human_review", "reporter": "reporter"},
    )
    graph.add_edge("human_review", "reporter")
    graph.add_edge("reporter", END)
    return graph


def compile_underwriting_graph(*, checkpointer: MemorySaver | None = None):
    saver = checkpointer or MemorySaver()
    return build_underwriting_graph().compile(checkpointer=saver)
