"""Strict Pydantic contracts for graph state and domain snapshots.

Trade-off vs LangGraph TypedDict state:
TypedDict + Annotated reducers are the framework default and make list
appends cheap. Pydantic is stricter (coercion, extra=forbid, JSON schema
for the reporter). We pick Pydantic because underwriting is a data-
integrity problem first: a silently dropped field is worse than a
slightly heavier merge.

True `frozen=True` on the *graph* state fights LangGraph's reducer model
(nodes return patches that the runtime assigns). Domain snapshots below
are frozen; AgentState is mutable-by-patch but closed to unknown keys.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PipelineStatus(str, Enum):
    PENDING = "PENDING"
    PARSED = "PARSED"
    VALIDATED = "VALIDATED"
    RISK_SCORED = "RISK_SCORED"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    NEED_MANUAL_INPUT = "NEED_MANUAL_INPUT"
    COMPLETE = "COMPLETE"


class Transaction(FrozenModel):
    txn_date: date
    description: str = Field(min_length=1)
    amount: float
    # Debit reduces cash; credit increases. Parser must not infer from sign
    # alone — mixed bank formats exist (positive-only amount + type column).
    direction: Literal["debit", "credit"]
    merchant: str | None = None
    category: str | None = None
    balance_after: float | None = None

    @field_validator("amount")
    @classmethod
    def amount_must_be_finite_non_negative(cls, value: float) -> float:
        if value < 0 or value != value:  # NaN check
            raise ValueError("amount must be a non-negative finite number")
        return round(value, 2)


class ParsedStatement(FrozenModel):
    source_path: str
    account_holder: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    currency: str = "INR"
    opening_balance: float | None = None
    closing_balance: float | None = None
    declared_monthly_income: float | None = None
    transactions: tuple[Transaction, ...] = ()
    parser_warnings: tuple[str, ...] = ()


class ValidationIssue(FrozenModel):
    code: str
    message: str
    severity: Literal["error", "warning"]


class ValidationResult(FrozenModel):
    is_valid: bool
    issues: tuple[ValidationIssue, ...] = ()
    # Canonical copy after type coercion / sort — never mutate the parser output.
    normalized_statement: ParsedStatement | None = None


class RiskFlag(FrozenModel):
    code: str
    detail: str
    weight: float = Field(ge=0, le=10)


class RiskAssessment(FrozenModel):
    dti_ratio: float | None = Field(default=None, ge=0)
    monthly_income: float | None = None
    monthly_debt_service: float | None = None
    risk_score: float = Field(ge=0, le=10)
    flags: tuple[RiskFlag, ...] = ()
    high_risk_merchants: tuple[str, ...] = ()
    rationale: str
    used_llm: bool
    tool_calls: tuple[str, ...] = ()


class HumanDecision(FrozenModel):
    approved: bool
    reviewer_notes: str = ""
    override_risk_score: float | None = Field(default=None, ge=0, le=10)


class AuditReport(FrozenModel):
    generated_at: datetime
    status: PipelineStatus
    source_path: str
    risk_assessment: RiskAssessment | None
    validation_issues: tuple[ValidationIssue, ...]
    human_decision: HumanDecision | None
    summary: str
    transaction_count: int


class AgentState(BaseModel):
    """Graph-level state. Nodes return partial dicts; LangGraph merges them."""

    model_config = ConfigDict(extra="forbid")

    source_path: str
    status: PipelineStatus = PipelineStatus.PENDING
    error_message: str | None = None
    parsed_statement: ParsedStatement | None = None
    validation: ValidationResult | None = None
    risk_assessment: RiskAssessment | None = None
    human_decision: HumanDecision | None = None
    report: dict[str, Any] | None = None
    # Append-only audit trail. Prefer tuples so callers cannot mutate in place.
    trace: tuple[str, ...] = ()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def append_trace(state: AgentState, event: str) -> tuple[str, ...]:
    return (*state.trace, event)
