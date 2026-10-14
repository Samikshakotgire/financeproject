"""Node 3 — DTI + anomaly flags, with a mandatory merchant-policy tool call.

Trade-off: DTI is computed in Python, not by the LLM. Ratio math is not a
language problem; letting GPT-4o-mini 'estimate' EMI totals is how
underwriting systems hallucinate a 0.31 instead of 0.62. The model is
used only to choose *which* merchant strings to look up via the policy
tool, then to write a short rationale. If the LLM/API is unavailable we
still emit a deterministic score so the pipeline remains demoable.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from underwriting_agent.config import get_chat_model, get_settings
from underwriting_agent.schemas import (
    AgentState,
    ParsedStatement,
    PipelineStatus,
    RiskAssessment,
    RiskFlag,
    append_trace,
)
from underwriting_agent.tools.merchant_risk import (
    HIGH_RISK_MERCHANT_CATEGORIES,
    lookup_high_risk_merchant_categories,
)

_DEBT_HINTS = re.compile(
    r"emi|loan|nbfc|credit\s*card|cc\s*payment|autodebit|ach|nach|housing|mortgage",
    re.I,
)
_INCOME_HINTS = re.compile(r"salary|payroll|neft.*salary|credit of salary", re.I)
_GAMBLE_HINTS = re.compile(r"bet|casino|stake|dream11|rummy", re.I)


def risk_evaluator_node(state: AgentState) -> dict:
    if state.status is PipelineStatus.NEED_MANUAL_INPUT:
        return {"trace": append_trace(state, "risk:skipped")}

    statement = (
        state.validation.normalized_statement
        if state.validation and state.validation.normalized_statement
        else state.parsed_statement
    )
    if statement is None:
        return {
            "status": PipelineStatus.NEED_MANUAL_INPUT,
            "error_message": "risk_evaluator: missing parsed statement",
            "trace": append_trace(state, "risk:fail:no_statement"),
        }

    dti_flags, dti_ratio, income, debt = _compute_dti(statement)
    anomaly_flags = _rule_anomalies(statement)
    tool_hits, tool_trace, used_llm, rationale = _merchant_tool_pass(statement)

    merchant_flags = [
        RiskFlag(code="HIGH_RISK_MCC", detail=hit, weight=2.0)
        for hit in tool_hits
    ]
    flags = tuple(dti_flags + anomaly_flags + merchant_flags)
    score = _score(dti_ratio, flags)

    assessment = RiskAssessment(
        dti_ratio=dti_ratio,
        monthly_income=income,
        monthly_debt_service=debt,
        risk_score=score,
        flags=flags,
        high_risk_merchants=tuple(tool_hits),
        rationale=rationale,
        used_llm=used_llm,
        tool_calls=tuple(tool_trace),
    )

    threshold = get_settings().human_review_risk_threshold
    next_status = (
        PipelineStatus.AWAITING_HUMAN if score > threshold else PipelineStatus.RISK_SCORED
    )
    return {
        "risk_assessment": assessment,
        "status": next_status,
        "trace": append_trace(state, f"risk:score={score}:status={next_status.value}"),
    }


def _compute_dti(statement: ParsedStatement) -> tuple[list[RiskFlag], float | None, float | None, float]:
    months = _span_months(statement)
    credits = [t for t in statement.transactions if t.direction == "credit"]
    salary_like = [t for t in credits if t.category == "salary" or _INCOME_HINTS.search(t.description)]
    income_pool = salary_like or credits
    total_income = sum(t.amount for t in income_pool)
    monthly_income = statement.declared_monthly_income or (total_income / months if months else None)

    debt_txns = [
        t
        for t in statement.transactions
        if t.direction == "debit" and (t.category in {"emi", "loan", "credit_card"} or _DEBT_HINTS.search(t.description))
    ]
    monthly_debt = (sum(t.amount for t in debt_txns) / months) if months else 0.0

    flags: list[RiskFlag] = []
    dti: float | None = None
    if monthly_income and monthly_income > 0:
        dti = round(monthly_debt / monthly_income, 4)
        if dti > 0.5:
            flags.append(RiskFlag(code="DTI_SEVERE", detail=f"DTI {dti:.2%} exceeds 50%", weight=4.0))
        elif dti > 0.4:
            flags.append(RiskFlag(code="DTI_ELEVATED", detail=f"DTI {dti:.2%} exceeds 40%", weight=2.5))
    else:
        flags.append(RiskFlag(code="INCOME_UNOBSERVED", detail="no salary/credit base for DTI", weight=3.0))

    return flags, dti, monthly_income, round(monthly_debt, 2)


def _rule_anomalies(statement: ParsedStatement) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    debits = [t for t in statement.transactions if t.direction == "debit"]
    if not debits:
        return flags

    amounts = [t.amount for t in debits]
    mean = sum(amounts) / len(amounts)
    variance = sum((a - mean) ** 2 for a in amounts) / len(amounts)
    stdev = math.sqrt(variance) if variance > 0 else 0.0

    for txn in debits:
        if stdev and txn.amount > mean + 3 * stdev:
            flags.append(
                RiskFlag(
                    code="OUTLIER_DEBIT",
                    detail=f"{txn.txn_date} {txn.description} {txn.amount}",
                    weight=1.5,
                )
            )
        if _GAMBLE_HINTS.search(txn.description):
            flags.append(RiskFlag(code="GAMBLING_NARRATION", detail=txn.description, weight=2.5))

    by_day: dict = defaultdict(float)
    for txn in debits:
        by_day[txn.txn_date] += txn.amount
    # Rapid-fire same-day cash-out is a structuring heuristic, not proof of fraud.
    if any(v > 200_000 for v in by_day.values()):
        flags.append(RiskFlag(code="HIGH_SAME_DAY_OUTFLOW", detail="> INR 2L debited in a single day", weight=2.0))
    return flags


def _merchant_tool_pass(statement: ParsedStatement) -> tuple[list[str], list[str], bool, str]:
    """Force a tool-calling round-trip against the policy list."""
    merchants = sorted(
        {
            (t.merchant or t.description)[:80]
            for t in statement.transactions
            if t.direction == "debit"
        }
    )
    payload = json.dumps(merchants[:25])

    try:
        model = get_chat_model().bind_tools(
            [lookup_high_risk_merchant_categories],
            tool_choice="lookup_high_risk_merchant_categories",
        )
        first = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a credit-risk analyst. You MUST call "
                        "lookup_high_risk_merchant_categories for the riskiest merchant "
                        "string in the list. Do not invent MCC codes."
                    )
                ),
                HumanMessage(content=payload),
            ]
        )
        tool_trace: list[str] = []
        hits: list[str] = []
        if getattr(first, "tool_calls", None):
            for call in first.tool_calls:
                result = lookup_high_risk_merchant_categories.invoke(call["args"])
                tool_trace.append(f"{call['name']}({call['args']}) -> {result}")
                if not str(result).startswith("no_policy_match") and result != "query_empty":
                    hits.extend(line.split(":", 1)[0] for line in str(result).splitlines())
            follow = get_chat_model().invoke(
                [
                    SystemMessage(content="Summarize residual merchant risk in 2 sentences."),
                    HumanMessage(content=payload),
                    first,
                    *[
                        ToolMessage(content=str(lookup_high_risk_merchant_categories.invoke(c["args"])), tool_call_id=c["id"])
                        for c in first.tool_calls
                    ],
                ]
            )
            rationale = str(follow.content)
            return hits, tool_trace, True, rationale
    except Exception as exc:
        # Deterministic fallback keeps the graph runnable without API keys.
        return _fallback_merchant_scan(statement, reason=str(exc))

    return _fallback_merchant_scan(statement, reason="model returned no tool_calls")


def _fallback_merchant_scan(statement: ParsedStatement, reason: str) -> tuple[list[str], list[str], bool, str]:
    hits: list[str] = []
    traces: list[str] = [f"fallback:{reason}"]
    for txn in statement.transactions:
        blob = f"{txn.merchant or ''} {txn.description} {txn.category or ''}".lower()
        for key in HIGH_RISK_MERCHANT_CATEGORIES:
            if key.replace("_", " ") in blob or key in blob:
                result = lookup_high_risk_merchant_categories.invoke({"query": key})
                traces.append(f"lookup_high_risk_merchant_categories({{'query': '{key}'}}) -> {result}")
                hits.append(key)
    return sorted(set(hits)), traces, False, f"Rule-based merchant scan ({reason})."


def _score(dti: float | None, flags: tuple[RiskFlag, ...] | list[RiskFlag]) -> float:
    base = 2.0
    if dti is not None:
        base += min(dti * 8.0, 5.0)
    weighted = sum(f.weight for f in flags)
    return round(min(10.0, base + weighted * 0.6), 2)


def _span_months(statement: ParsedStatement) -> float:
    if statement.period_start and statement.period_end:
        days = max((statement.period_end - statement.period_start).days, 1)
        return max(days / 30.437, 1.0)
    dates = [t.txn_date for t in statement.transactions]
    if not dates:
        return 1.0
    days = max((max(dates) - min(dates)).days, 1)
    return max(days / 30.437, 1.0)
