"""Run an audit from the command line."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from langgraph.types import Command

from underwriting_agent.graph import compile_underwriting_graph
from underwriting_agent.schemas import AgentState


def main() -> None:
    parser = argparse.ArgumentParser(description="Bank-statement underwriting audit")
    parser.add_argument("source", type=Path, help="Path to a .csv or .pdf statement")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument(
        "--resume-json",
        default=None,
        help="HumanDecision JSON to resume a paused high-risk run",
    )
    args = parser.parse_args()

    graph = compile_underwriting_graph()
    thread_id = args.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    if args.resume_json:
        decision = json.loads(Path(args.resume_json).read_text(encoding="utf-8"))
        result = graph.invoke(Command(resume=decision), config)
    else:
        result = graph.invoke(
            AgentState(source_path=str(args.source.resolve())),
            config,
        )

    print(json.dumps(_public(result), indent=2, default=str))
    print(f"\nthread_id={thread_id}")


def _public(result: AgentState | dict) -> dict:
    if isinstance(result, AgentState):
        payload = result.model_dump(mode="json")
    else:
        payload = dict(result)
        if "report" not in payload and hasattr(result, "report"):
            payload = result
    return {
        "status": getattr(result, "status", payload.get("status")),
        "error_message": getattr(result, "error_message", payload.get("error_message")),
        "report": getattr(result, "report", None) or payload.get("report"),
        "trace": getattr(result, "trace", None) or payload.get("trace"),
    }


if __name__ == "__main__":
    main()
