"""
Canonical GKX characteristic registry (94 variables).

This module defines:
1. The full GKX variable name universe.
2. Metadata describing proxy status and expected inputs.
3. Helpers used by the pipeline and validation checks.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


GKX_94: List[str] = [
    "absacc", "acc", "aeavol", "age", "agr", "baspread", "beta", "betasq", "bm", "bm_ia",
    "cash", "cashdebt", "cashpr", "cfp", "cfp_ia", "chatoia", "chcsho", "chempia", "chinv",
    "chmom", "chpmia", "chtx", "cinvest", "convind", "currat", "depr", "divi", "divo",
    "dolvol", "dy", "ear", "egr", "ep", "gma", "grCAPX", "grltnoa", "herf", "hire", "idiovol",
    "ill", "indmom", "invest", "lev", "lgr", "maxret", "mom12m", "mom1m", "mom36m", "mom6m",
    "ms", "mvel1", "mve_ia", "nincr", "operprof", "orgcap", "pchcapx_ia", "pchcurrat",
    "pchdepr", "pchgm_pchsale", "pchquick", "pchsale_pchinvt", "pchsale_pchrect",
    "pchsale_pchxsga", "pchsaleinv", "pctacc", "pricedelay", "ps", "quick", "rd", "rd_mve",
    "rd_sale", "realestate", "retvol", "roaq", "roavol", "roeq", "roic", "rsup", "salecash",
    "saleinv", "salerec", "secured", "securedind", "sgr", "sin", "sp", "std_dolvol",
    "std_turn", "stdacc", "stdcf", "tang", "tb", "turn", "zerotrade",
]


_ALIASES: Dict[str, str] = {
    "absacc": "Absolute accruals",
    "acc": "Accruals",
    "aeavol": "Abnormal earnings announcement volume proxy",
    "age": "Listing age proxy",
    "agr": "Asset growth",
    "baspread": "Bid-ask spread proxy",
    "beta": "Market beta",
    "betasq": "Squared beta",
    "bm": "Book-to-market",
    "bm_ia": "Industry-adjusted book-to-market",
    "cash": "Cash-to-assets",
    "cashdebt": "Cash-to-debt",
    "cashpr": "Cash productivity proxy",
    "cfp": "Cash-flow-to-price proxy",
    "cfp_ia": "Industry-adjusted cash-flow-to-price proxy",
    "chatoia": "Change in asset turnover (industry-adjusted proxy)",
    "chcsho": "Change in shares outstanding",
    "chempia": "Change in employee productivity proxy",
    "chinv": "Change in inventory",
    "chmom": "Change in momentum",
    "chpmia": "Change in profit margin (industry-adjusted proxy)",
    "chtx": "Change in effective tax",
    "cinvest": "Corporate investment proxy",
    "convind": "Convertible debt indicator",
    "currat": "Current ratio",
    "depr": "Depreciation ratio",
    "divi": "Dividend initiation indicator",
    "divo": "Dividend omission indicator",
    "dolvol": "Dollar volume",
    "dy": "Dividend yield",
    "ear": "Earnings announcement return proxy",
    "egr": "Equity growth proxy",
    "ep": "Earnings-to-price",
    "gma": "Gross profitability",
    "grCAPX": "Growth in capital expenditures",
    "grltnoa": "Growth in long-term net operating assets",
    "herf": "Industry Herfindahl concentration",
    "hire": "Hiring proxy",
    "idiovol": "Idiosyncratic volatility proxy",
    "ill": "Illiquidity proxy",
    "indmom": "Industry momentum",
    "invest": "Investment proxy",
    "lev": "Leverage",
    "lgr": "Liability growth proxy",
    "maxret": "Maximum daily return in month",
    "mom12m": "12-month momentum",
    "mom1m": "1-month momentum",
    "mom36m": "36-month momentum",
    "mom6m": "6-month momentum",
    "ms": "Share issuance/financing proxy",
    "mvel1": "Log market equity",
    "mve_ia": "Industry-adjusted log market equity",
    "nincr": "Consecutive earnings increases proxy",
    "operprof": "Operating profitability",
    "orgcap": "Organizational capital proxy",
    "pchcapx_ia": "Industry-adjusted change in capex",
    "pchcurrat": "Change in current ratio",
    "pchdepr": "Change in depreciation ratio",
    "pchgm_pchsale": "Change in gross margin minus change in sales",
    "pchquick": "Change in quick ratio",
    "pchsale_pchinvt": "Change in sales minus change in inventory",
    "pchsale_pchrect": "Change in sales minus change in receivables",
    "pchsale_pchxsga": "Change in sales minus change in SG&A",
    "pchsaleinv": "Change in sales-to-inventory",
    "pctacc": "Percent accruals",
    "pricedelay": "Price delay proxy",
    "ps": "Price-to-sales",
    "quick": "Quick ratio",
    "rd": "R&D intensity proxy",
    "rd_mve": "R&D-to-market-equity",
    "rd_sale": "R&D-to-sales",
    "realestate": "Real-estate intensity proxy",
    "retvol": "Return volatility",
    "roaq": "Return on assets (quarterly proxy)",
    "roavol": "ROA volatility proxy",
    "roeq": "Return on equity (quarterly proxy)",
    "roic": "Return on invested capital proxy",
    "rsup": "Revenue surprise proxy",
    "salecash": "Sales-to-cash",
    "saleinv": "Sales-to-inventory",
    "salerec": "Sales-to-receivables",
    "secured": "Secured debt ratio proxy",
    "securedind": "Secured debt indicator proxy",
    "sgr": "Sales growth",
    "sin": "Sin stock indicator proxy",
    "sp": "Sales-to-price",
    "std_dolvol": "Std. dev. of dollar volume",
    "std_turn": "Std. dev. of turnover",
    "stdacc": "Std. dev. of accruals proxy",
    "stdcf": "Std. dev. of cashflow proxy",
    "tang": "Asset tangibility",
    "tb": "Tax burden proxy",
    "turn": "Turnover",
    "zerotrade": "Zero-trade days fraction",
}


_DIRECT_PROXY = {
    "agr", "baspread", "beta", "betasq", "bm", "cash", "cashdebt", "cfp", "chcsho", "chinv",
    "chmom", "currat", "depr", "dolvol", "dy", "ep", "gma", "grCAPX", "herf", "idiovol",
    "ill", "indmom", "invest", "lev", "lgr", "maxret", "mom12m", "mom1m", "mom36m", "mom6m",
    "mvel1", "operprof", "pchcurrat", "pchdepr", "pchquick", "pchsale_pchinvt",
    "pchsale_pchrect", "pchsale_pchxsga", "pctacc", "ps", "quick", "rd", "rd_mve", "rd_sale",
    "realestate", "retvol", "roaq", "roavol", "roeq", "roic", "salecash", "saleinv", "salerec",
    "secured", "securedind", "sgr", "sin", "sp", "std_dolvol", "std_turn", "tang", "tb", "turn",
    "zerotrade",
}

_WEAK_PROXY = {
    "absacc", "acc", "aeavol", "age", "bm_ia", "cashpr", "cfp_ia", "chatoia", "chempia",
    "chpmia", "chtx", "cinvest", "convind", "divi", "divo", "ear", "egr", "grltnoa", "hire",
    "ms", "mve_ia", "nincr", "orgcap", "pchcapx_ia", "pchgm_pchsale", "pchsaleinv",
    "pricedelay", "rsup", "stdacc", "stdcf",
}


def _status_for(name: str) -> str:
    if name in _DIRECT_PROXY:
        return "direct_proxy"
    if name in _WEAK_PROXY:
        return "weak_proxy"
    return "nan_fallback"


def get_gkx_feature_names() -> List[str]:
    """Return the canonical GKX variable names."""
    return list(GKX_94)


def get_gkx_alias_map() -> Dict[str, str]:
    """Return acronym -> human readable alias."""
    return dict(_ALIASES)


def get_gkx_registry() -> pd.DataFrame:
    """Return registry as a DataFrame for reporting/export."""
    rows = []
    for name in GKX_94:
        rows.append({
            "gkx_name": name,
            "alias": _ALIASES.get(name, name),
            "frequency": "monthly",
            "status": _status_for(name),
        })
    return pd.DataFrame(rows)


def get_gkx_schema_check(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Validate presence of all canonical columns.

    Returns:
        Dict with keys: present, missing
    """
    cols = set(df.columns)
    present = [c for c in GKX_94 if c in cols]
    missing = [c for c in GKX_94 if c not in cols]
    return {"present": present, "missing": missing}

