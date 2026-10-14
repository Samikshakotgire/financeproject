"""Node 2 — schema + ledger integrity.

Trade-off: we validate *after* parse, not inside pdfplumber loops. Mixing
extraction with Pydantic raises would make it hard to tell 'PDF is garbage'
from 'one row has a bad date'. Parser failures already go to NEED_MANUAL_INPUT;
validator errors do the same because a partially valid statement is more
dangerous in credit than a hard stop.
"""

from __future__ import annotations

from underwriting_agent.schemas import (
    AgentState,
    ParsedStatement,
    PipelineStatus,
    Transaction,
    ValidationIssue,
    ValidationResult,
    append_trace,
)


def validator_node(state: AgentState) -> dict:
    if state.status is PipelineStatus.NEED_MANUAL_INPUT or state.parsed_statement is None:
        return {"trace": append_trace(state, "validator:skipped")}

    issues: list[ValidationIssue] = []
    statement = state.parsed_statement

    if not statement.transactions:
        issues.append(ValidationIssue(code="EMPTY_LEDGER", message="no transactions", severity="error"))

    issues.extend(_check_date_order(statement.transactions))
    issues.extend(_check_balance_continuity(statement))
    issues.extend(_check_duplicate_rows(statement.transactions))

    errors = tuple(i for i in issues if i.severity == "error")
    is_valid = len(errors) == 0
    # Re-validate through Pydantic so coerced types are the ones risk sees.
    normalized = ParsedStatement.model_validate(statement.model_dump())

    validation = ValidationResult(
        is_valid=is_valid,
        issues=tuple(issues),
        normalized_statement=normalized if is_valid else None,
    )

    if not is_valid:
        return {
            "validation": validation,
            "status": PipelineStatus.NEED_MANUAL_INPUT,
            "error_message": "validation_failed: " + "; ".join(e.message for e in errors),
            "trace": append_trace(state, f"validator:fail:{len(errors)} errors"),
        }

    return {
        "validation": validation,
        "parsed_statement": normalized,
        "status": PipelineStatus.VALIDATED,
        "trace": append_trace(state, f"validator:ok:{len(issues)} warnings"),
    }


def _check_date_order(txns: tuple[Transaction, ...]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    previous = None
    for txn in txns:
        if previous and txn.txn_date < previous:
            issues.append(
                ValidationIssue(
                    code="DATE_OUT_OF_ORDER",
                    message=f"{txn.txn_date} precedes {previous}",
                    severity="warning",
                )
            )
        previous = txn.txn_date
    return issues


def _check_balance_continuity(statement: ParsedStatement) -> list[ValidationIssue]:
    """If running balances exist, they must match amount + direction.

    Why: a 1-rupee break is a stronger fraud/OCR signal than any LLM summary.
    """
    issues: list[ValidationIssue] = []
    with_balance = [t for t in statement.transactions if t.balance_after is not None]
    if len(with_balance) < 2:
        return issues

    expected = with_balance[0].balance_after
    for txn in with_balance[1:]:
        delta = txn.amount if txn.direction == "credit" else -txn.amount
        expected = round((expected or 0) + delta, 2)
        if txn.balance_after is not None and abs(txn.balance_after - expected) > 0.05:
            issues.append(
                ValidationIssue(
                    code="BALANCE_BREAK",
                    message=f"{txn.txn_date} {txn.description}: expected {expected}, saw {txn.balance_after}",
                    severity="error",
                )
            )
            expected = txn.balance_after
    return issues


def _check_duplicate_rows(txns: tuple[Transaction, ...]) -> list[ValidationIssue]:
    seen: set[tuple] = set()
    issues: list[ValidationIssue] = []
    for txn in txns:
        key = (txn.txn_date, txn.description, txn.amount, txn.direction)
        if key in seen:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_TXN",
                    message=f"duplicate {key}",
                    severity="warning",
                )
            )
        seen.add(key)
    return issues
