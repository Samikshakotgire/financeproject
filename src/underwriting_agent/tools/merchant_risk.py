"""Policy lookup for high-risk merchant categories.

Trade-off: this is a tool, not a prompt dump. Policy lists change with
compliance reviews; stuffing them into the system prompt would require a
model redeploy (and leak the full list into every token bill). A tool also
makes the graph auditable: we persist the exact query/result on RiskAssessment.
"""

from __future__ import annotations

from langchain_core.tools import tool

# MCC-style buckets used by typical Indian acquiring / BNPL underwriting desks.
# Values are illustrative policy labels, not a live bureau feed.
HIGH_RISK_MERCHANT_CATEGORIES: dict[str, str] = {
    "gambling": "Online/offline betting, casinos, lottery",
    "crypto_exchange": "Cryptocurrency exchanges and OTC desks",
    "payday_lending": "Short-term high-APR lenders",
    "money_transfer": "P2P remittance / hawala-like corridors",
    "adult": "Adult entertainment merchants",
    "cash_advance": "ATM-style cash-out at merchants",
    "unlicensed_nbfc": "Non-bank credit without a visible RBI registration cue",
    "gift_card_reseller": "Stored-value resellers used for layering",
}


@tool
def lookup_high_risk_merchant_categories(query: str) -> str:
    """Query the underwriting policy list of high-risk merchant categories.

    Pass a free-text merchant description or a category key. Returns matching
    policy rows so the risk node can flag exposures without hallucinating MCC codes.
    """
    q = query.lower().strip()
    if not q:
        return "query_empty"

    hits: list[str] = []
    for key, description in HIGH_RISK_MERCHANT_CATEGORIES.items():
        haystack = f"{key} {description}".lower()
        if q in haystack or key in q or any(token in haystack for token in q.split() if len(token) > 3):
            hits.append(f"{key}: {description}")

    if not hits:
        return f"no_policy_match:{query}"
    return "\n".join(hits)
