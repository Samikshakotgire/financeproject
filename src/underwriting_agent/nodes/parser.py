"""Node 1 — extract a statement from PDF or CSV.

Trade-off: pdfplumber over OCR. Bank PDFs in India are usually digital
text layers; OCR would add latency and digit-swap errors (8 vs 0) that
silently corrupt balances. If this parser cannot recover a ledger, we
do *not* guess — we fail closed to NEED_MANUAL_INPUT.
"""

from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path

import pdfplumber

from underwriting_agent.schemas import (
    AgentState,
    ParsedStatement,
    PipelineStatus,
    Transaction,
    append_trace,
)

_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d-%b-%Y")
_DEBIT_HINTS = ("debit", "dr", "withdrawal", "paid", "purchase")
_CREDIT_HINTS = ("credit", "cr", "deposit", "salary", "interest")


def parser_node(state: AgentState) -> dict:
    path = Path(state.source_path)
    try:
        if not path.exists():
            raise FileNotFoundError(f"statement not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            statement = _parse_csv(path)
        elif suffix == ".pdf":
            statement = _parse_pdf(path)
        else:
            raise ValueError(f"unsupported statement format: {suffix}")

        if not statement.transactions:
            raise ValueError("parser produced zero transactions")

        return {
            "parsed_statement": statement,
            "status": PipelineStatus.PARSED,
            "error_message": None,
            "trace": append_trace(state, f"parser:ok:{len(statement.transactions)} txns"),
        }
    except Exception as exc:
        # Fail closed. Downstream nodes must not invent cashflows.
        return {
            "parsed_statement": None,
            "status": PipelineStatus.NEED_MANUAL_INPUT,
            "error_message": f"parse_failed: {exc}",
            "trace": append_trace(state, f"parser:fail:{exc}"),
        }


def _parse_csv(path: Path) -> ParsedStatement:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        rows = list(reader)

    txns = tuple(_row_to_txn(row, idx) for idx, row in enumerate(rows, start=2))
    income = _optional_float(_first(rows, "declared_monthly_income"))
    return ParsedStatement(
        source_path=str(path),
        account_holder=_first(rows, "account_holder"),
        currency=_first(rows, "currency") or "INR",
        declared_monthly_income=income,
        opening_balance=_optional_float(_first(rows, "opening_balance")),
        closing_balance=_optional_float(_first(rows, "closing_balance")),
        transactions=txns,
        parser_warnings=(),
        period_start=min((t.txn_date for t in txns), default=None),
        period_end=max((t.txn_date for t in txns), default=None),
    )


def _parse_pdf(path: Path) -> ParsedStatement:
    warnings: list[str] = []
    extracted_rows: list[dict[str, str]] = []
    account_holder: str | None = None

    with pdfplumber.open(path) as pdf:
        if not pdf.pages:
            raise ValueError("PDF has no pages")
        for page in pdf.pages:
            text = page.extract_text() or ""
            if account_holder is None:
                account_holder = _extract_holder(text)
            tables = page.extract_tables() or []
            if not tables:
                warnings.append(f"no table on page {page.page_number}; falling back to line regex")
                extracted_rows.extend(_regex_lines(text))
                continue
            for table in tables:
                extracted_rows.extend(_table_to_rows(table))

    if not extracted_rows:
        raise ValueError("no tabular or regex-matched transactions in PDF")

    txns = tuple(_row_to_txn(row, idx) for idx, row in enumerate(extracted_rows, start=1))
    return ParsedStatement(
        source_path=str(path),
        account_holder=account_holder,
        transactions=txns,
        parser_warnings=tuple(warnings),
        period_start=min((t.txn_date for t in txns), default=None),
        period_end=max((t.txn_date for t in txns), default=None),
    )


def _row_to_txn(row: dict[str, str], line_no: int) -> Transaction:
    normalized = { (k or "").strip().lower().replace(" ", "_"): (v or "").strip() for k, v in row.items() }
    raw_date = normalized.get("date") or normalized.get("txn_date") or normalized.get("transaction_date")
    desc = normalized.get("description") or normalized.get("narration") or normalized.get("particulars")
    if not raw_date or not desc:
        raise ValueError(f"line {line_no}: missing date or description")

    amount, direction = _amount_and_direction(normalized)
    return Transaction(
        txn_date=_parse_date(raw_date),
        description=desc,
        amount=amount,
        direction=direction,
        merchant=normalized.get("merchant") or None,
        category=normalized.get("category") or None,
        balance_after=_optional_float(normalized.get("balance") or normalized.get("balance_after")),
    )


def _amount_and_direction(row: dict[str, str]) -> tuple[float, str]:
    debit = _optional_float(row.get("debit") or row.get("withdrawal"))
    credit = _optional_float(row.get("credit") or row.get("deposit"))
    if debit and credit:
        raise ValueError("row has both debit and credit amounts")
    if debit:
        return debit, "debit"
    if credit:
        return credit, "credit"

    amount = _optional_float(row.get("amount"))
    if amount is None:
        raise ValueError("row has no amount")
    explicit = (row.get("type") or row.get("dr_cr") or row.get("direction") or "").lower()
    if any(h in explicit for h in _DEBIT_HINTS):
        return amount, "debit"
    if any(h in explicit for h in _CREDIT_HINTS):
        return amount, "credit"
    raise ValueError("cannot determine debit vs credit without a type column")


def _parse_date(raw: str) -> date:
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date: {raw}")


def _optional_float(raw: str | None) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    cleaned = re.sub(r"[₹,\s]", "", str(raw))
    return float(cleaned)


def _first(rows: list[dict[str, str]], key: str) -> str | None:
    for row in rows:
        for k, v in row.items():
            if (k or "").strip().lower().replace(" ", "_") == key and v:
                return v.strip()
    return None


def _extract_holder(text: str) -> str | None:
    match = re.search(r"(?:account holder|name)\s*[:\-]\s*(.+)", text, re.I)
    return match.group(1).strip() if match else None


def _table_to_rows(table: list[list[str | None]]) -> list[dict[str, str]]:
    if not table or len(table) < 2:
        return []
    headers = [(cell or f"col_{i}").strip().lower().replace(" ", "_") for i, cell in enumerate(table[0])]
    rows: list[dict[str, str]] = []
    for raw in table[1:]:
        if not raw or all(not (c or "").strip() for c in raw):
            continue
        rows.append({headers[i]: (raw[i] or "").strip() for i in range(min(len(headers), len(raw)))})
    return rows


def _regex_lines(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(
        r"(?P<date>\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+(?P<desc>.+?)\s+(?P<amount>[\d,]+\.\d{2})\s+(?P<type>Dr|Cr|DR|CR)"
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "date": match.group("date"),
                "description": match.group("desc").strip(),
                "amount": match.group("amount"),
                "type": match.group("type"),
            }
        )
    return rows
