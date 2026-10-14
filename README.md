# Autonomous Financial Underwriting & Audit Agent

LangGraph pipeline that audits a bank statement the way a credit-ops desk would: **parse → validate → score risk → (optional human) → structured JSON**. Built as a production-shaped skeleton for a fintech agentic-workflow role — data integrity over UI.

```
PDF/CSV  →  Parser  →  Validator  →  Risk_Evaluator  →  Reporter → JSON
                                      │
                                      └─ if risk_score > 7 → Human-in-the-loop interrupt
```

## Why this shape

| Choice | Trade-off |
|---|---|
| **LangGraph nodes, not one mega-prompt** | Each stage has a typed contract. Failures are local (parse vs risk) instead of a single hallucinated narrative. |
| **Pydantic state** | Stricter than TypedDict. Slightly heavier merges; worth it because a dropped `direction` field is a credit bug. |
| **DTI in Python, merchants via LLM tool** | Ratios must not be estimated by a language model. GPT-4o-mini only *selects* which merchant string to look up against policy. |
| **pdfplumber, not OCR** | Digital bank PDFs have a text layer. OCR introduces digit swaps that break running balances. |
| **Fail closed → `NEED_MANUAL_INPUT`** | Partial ledgers are more dangerous than a hard stop. |
| **`interrupt()` for HITL** | Durable checkpoint instead of a custom webhook. Swap `MemorySaver` for Postgres when you have an ops UI. |
| **Model factory** | Default `gpt-4o-mini`; `MODEL_PROVIDER=ollama` for a local swap with no node changes. Temperature 0 for reproducible scores. |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

Set `OPENAI_API_KEY` if you want the live tool-calling pass. Without a key the risk node **falls back to a deterministic merchant scan** so the graph still runs.

Local model later:

```
MODEL_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

## Run

```bash
python -m underwriting_agent.cli examples/sample_statement.csv
python -m underwriting_agent.cli examples/high_risk_statement.csv
```

High-risk runs pause after `Risk_Evaluator`. Resume with a `HumanDecision`:

```bash
python -m underwriting_agent.cli examples/high_risk_statement.csv --thread-id demo-1
python -m underwriting_agent.cli examples/high_risk_statement.csv --thread-id demo-1 --resume-json examples/resume_approve.json
```

```bash
pytest
```

## Modules

| Path | Role |
|---|---|
| `schemas.py` | Frozen domain models + `AgentState` |
| `nodes/parser.py` | pdfplumber / CSV → `ParsedStatement` |
| `nodes/validator.py` | Schema + running-balance integrity |
| `nodes/risk_evaluator.py` | DTI, anomalies, **required** `lookup_high_risk_merchant_categories` tool |
| `nodes/human_review.py` | LangGraph `interrupt()` when score > 7 |
| `nodes/reporter.py` | JSON `AuditReport` |
| `config.py` | Swappable chat model |
| `graph.py` | Compilation + routing |

## Policy notes

- DTI = observed monthly debt service (EMI / loan / card) ÷ salary-like credits (or `declared_monthly_income` if present).
- High-risk MCC list lives in `tools/merchant_risk.py` so compliance can edit policy without touching prompts.
- Risk score is a capped 0–10 blend of DTI and flag weights — explicit and replaceable, not a black-box ranker.
