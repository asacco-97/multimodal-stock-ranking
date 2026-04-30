"""
Build the monthly modeling base table used by notebook 003a_gbm_model_refactor.

This script intentionally does NOT create cross-sectional transforms. Those are
computed split-locally during modeling to avoid leakage.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features.gkx_registry import GKX_94


DEFAULT_MACRO_COLS = [
    "DFF",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "CPIAUCSL",
    "CPILFESL",
    "PCEPI",
    "GDP",
    "GDPC1",
    "INDPRO",
    "UNRATE",
    "PAYEMS",
    "ICSA",
    "UMCSENT",
    "RSXFS",
    "VIXCLS",
    "DCOILWTICO",
    "M2SL",
    "TOTALSL",
]

# By default, interact SIC2 dummies with the raw monthly macro block.
DEFAULT_MACRO_INTERACTION_COLS = list(DEFAULT_MACRO_COLS)

SIC_SOURCE_COLS = ("sic2", "sic", "sic_code", "sic_cd", "SIC", "SICCD")


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _load_ff_risk_free(ff_factors_path: str) -> pd.DataFrame:
    """
    Load Fama-French factors and return month_end + RF.
    Matches notebook behavior.
    """
    ff = pd.read_csv(ff_factors_path, skiprows=3)
    ff.columns = ["year_month"] + list(ff.columns[1:])
    ff["year_month"] = pd.to_numeric(ff["year_month"], errors="coerce")
    ff = ff.dropna(subset=["year_month"])
    ff["ymd_int"] = (ff["year_month"].astype(int).astype(str) + "01").astype(int)
    ff = ff.query("ymd_int > 999999")
    ff["month_start"] = pd.to_datetime(ff["ymd_int"], format="%Y%m%d")
    ff["month_end"] = ff["month_start"] + pd.offsets.MonthEnd(0)
    ff["RF"] = pd.to_numeric(ff["RF"], errors="coerce")
    ff["RF"] = ff["RF"] / 100  # Convert from percentage to decimal
    return ff[["month_end", "RF"]].drop_duplicates("month_end")


def _load_etf_flags(universe_file: Optional[str]) -> Optional[pd.DataFrame]:
    if not universe_file:
        return None
    if not os.path.exists(universe_file):
        raise FileNotFoundError(f"Universe file not found: {universe_file}")
    ext = os.path.splitext(universe_file)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(universe_file)
        if {"ticker", "is_etf"}.issubset(df.columns):
            out = df[["ticker", "is_etf"]].copy()
            out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
            return out
    return None


def _series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=np.float64)


def _safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    out = series.pct_change(periods=periods)
    return out.replace([np.inf, -np.inf], np.nan)


def _build_monthly_macro_snapshot(
    daily: pd.DataFrame, macro_cols: Sequence[str]
) -> Tuple[pd.DataFrame, List[str]]:
    macro_available = [c for c in macro_cols if c in daily.columns]
    if not macro_available:
        return pd.DataFrame({"month_end": sorted(daily["month_end"].dropna().unique())}), []

    macro_monthly = (
        daily.sort_values("date")
        .groupby("month_end", as_index=False)
        .tail(1)[["month_end"] + macro_available]
        .drop_duplicates("month_end")
        .sort_values("month_end")
        .reset_index(drop=True)
    )
    return macro_monthly, macro_available


def _add_macro_derived_features(
    macro_monthly: pd.DataFrame, macro_available: Sequence[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create monthly macro transforms and a paper-style 8-variable macro block.
    """
    if not macro_available:
        return macro_monthly, []

    out = macro_monthly.copy()
    derived_cols: List[str] = []

    for col in macro_available:
        base = pd.to_numeric(out[col], errors="coerce")
        chg_col = f"{col}_chg1m"
        pct1_col = f"{col}_pct1m"
        pct12_col = f"{col}_pct12m"
        out[chg_col] = base.diff(1)
        out[pct1_col] = _safe_pct_change(base, 1)
        out[pct12_col] = _safe_pct_change(base, 12)
        derived_cols.extend([chg_col, pct1_col, pct12_col])

    dff = _series_or_nan(out, "DFF")
    dgs10 = _series_or_nan(out, "DGS10")
    dgs2 = _series_or_nan(out, "DGS2")
    t10y2y = _series_or_nan(out, "T10Y2Y")
    if t10y2y.isna().all():
        t10y2y = dgs10 - dgs2

    out["macro_rate_dff"] = dff
    out["macro_term_spread"] = t10y2y
    out["macro_cpi_yoy"] = _safe_pct_change(_series_or_nan(out, "CPIAUCSL"), 12)
    out["macro_indpro_yoy"] = _safe_pct_change(_series_or_nan(out, "INDPRO"), 12)
    out["macro_unrate"] = _series_or_nan(out, "UNRATE")
    out["macro_vix"] = _series_or_nan(out, "VIXCLS")
    out["macro_oil_yoy"] = _safe_pct_change(_series_or_nan(out, "DCOILWTICO"), 12)
    out["macro_m2_yoy"] = _safe_pct_change(_series_or_nan(out, "M2SL"), 12)

    paper_style_cols = [
        "macro_rate_dff",
        "macro_term_spread",
        "macro_cpi_yoy",
        "macro_indpro_yoy",
        "macro_unrate",
        "macro_vix",
        "macro_oil_yoy",
        "macro_m2_yoy",
    ]
    derived_cols.extend(paper_style_cols)

    for col in derived_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    macro_feature_cols = _dedupe_keep_order(list(macro_available) + derived_cols)
    return out, macro_feature_cols


def _coerce_sic2(series: pd.Series) -> pd.Series:
    def _to_sic2(value: Any) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if not digits:
            return np.nan
        if len(digits) >= 2:
            return float(int(digits[:2]))
        return float(int(digits))

    return series.map(_to_sic2).astype(np.float64)


def _last_valid_or_nan(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(valid.iloc[-1])


def _build_ticker_sic2_map(daily: pd.DataFrame) -> pd.DataFrame:
    sic_cols = [c for c in SIC_SOURCE_COLS if c in daily.columns]
    if not sic_cols:
        return pd.DataFrame(columns=["ticker", "sic2"])

    tmp = daily[["ticker"] + sic_cols].copy()
    for col in sic_cols:
        tmp[col] = _coerce_sic2(tmp[col])
    tmp["sic2"] = tmp[sic_cols].bfill(axis=1).iloc[:, 0]

    ticker_sic2 = (
        tmp.groupby("ticker", as_index=False)["sic2"]
        .agg(_last_valid_or_nan)
        .rename(columns={"sic2": "sic2"})
    )
    return ticker_sic2


def _build_sic2_dummies(sic2: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
    labels = sic2.map(lambda x: f"{int(x):02d}" if np.isfinite(x) else "missing")
    dummies = pd.get_dummies(labels, prefix="sic2", dtype=np.float32)
    return dummies, list(dummies.columns)


def _build_macro_sic_interactions(
    df: pd.DataFrame,
    macro_cols: Sequence[str],
    sic2_col: str = "sic2",
    top_n_sic2: Optional[int] = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    if not macro_cols or sic2_col not in df.columns:
        return pd.DataFrame(index=df.index), []

    sic2_numeric = pd.to_numeric(df[sic2_col], errors="coerce")
    sic2_freq = sic2_numeric.dropna()
    if sic2_freq.empty:
        return pd.DataFrame(index=df.index), []

    sic2_counts = sic2_freq.astype(np.int64).value_counts()
    if top_n_sic2 is not None and top_n_sic2 > 0:
        sic2_counts = sic2_counts.head(top_n_sic2)
    sic2_values = sorted(int(v) for v in sic2_counts.index if np.isfinite(v))
    if not sic2_values:
        return pd.DataFrame(index=df.index), []

    macro_block = df[list(macro_cols)].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    frames: List[pd.DataFrame] = []
    names: List[str] = []
    for sic2_value in sic2_values:
        sic2_label = f"{sic2_value:02d}"
        mask = (sic2_numeric == sic2_value).astype(np.float32)
        interaction = macro_block.mul(mask, axis=0)
        interaction.columns = [f"{sic2_label}__x__{macro_col}" for macro_col in macro_cols]
        frames.append(interaction)
        names.extend(interaction.columns.tolist())

    return pd.concat(frames, axis=1), names


def build_model_df(
    gkx_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    ff_factors_path: str,
    universe_file: Optional[str] = None,
    macro_cols: Optional[List[str]] = None,
    macro_interaction_cols: Optional[List[str]] = None,
    include_macro_sic_interactions: bool = True,
) -> pd.DataFrame:
    macro_cols = macro_cols or DEFAULT_MACRO_COLS
    ff = _load_ff_risk_free(ff_factors_path)

    daily = daily_df.copy()
    daily = daily.drop(columns=[c for c in daily.columns if c.endswith("_x")], errors="ignore")
    daily.columns = [c.replace("_y", "") for c in daily.columns]
    daily["date"] = pd.to_datetime(daily["date"])
    daily["month_end"] = daily["date"] + pd.offsets.MonthEnd(0)
    daily = daily.merge(ff, on="month_end", how="left")

    if "close" not in daily.columns:
        raise ValueError("daily_df must include a lowercase 'close' column.")

    px_m = (
        daily.sort_values(["ticker", "date"])
        .groupby(["ticker", "month_end"], as_index=False)
        .tail(1)[["ticker", "month_end", "RF", "close"]]
    )
    px_m = px_m.sort_values(["ticker", "month_end"])
    px_m["ret_eom_t+1"] = px_m.groupby("ticker")["close"].shift(-1) / px_m["close"] - 1
    px_m["ret_eom_t+1_excess"] = px_m["ret_eom_t+1"] - px_m.groupby("ticker")["RF"].shift(-1)

    df_spy = px_m.loc[px_m["ticker"] == "SPY", ["month_end", "ret_eom_t+1"]].rename(
        columns={"ret_eom_t+1": "spy_ret_eom_t+1"}
    )

    macro_monthly, macro_available = _build_monthly_macro_snapshot(daily, macro_cols)
    macro_monthly, macro_feature_cols = _add_macro_derived_features(macro_monthly, macro_available)

    df = gkx_df.merge(
        px_m[["ticker", "month_end", "RF", "ret_eom_t+1", "ret_eom_t+1_excess"]],
        on=["ticker", "month_end"],
        how="left",
    )
    df = df.merge(df_spy, on="month_end", how="left")
    if macro_feature_cols:
        df = df.merge(macro_monthly[["month_end"] + macro_feature_cols], on="month_end", how="left")

    ticker_sic2 = _build_ticker_sic2_map(daily)
    if not ticker_sic2.empty:
        df = df.merge(ticker_sic2, on="ticker", how="left")
    else:
        df["sic2"] = np.nan
    df["sic2"] = _coerce_sic2(df["sic2"])
    if df["sic2"].notna().sum() == 0:
        print("Warning: sic2 is fully missing; macro x SIC2 interactions will not be generated.")

    sic_dummies, sic_dummy_cols = _build_sic2_dummies(df["sic2"])
    if sic_dummy_cols:
        df = pd.concat([df, sic_dummies], axis=1)

    interaction_cols: List[str] = []
    interaction_base_cols = macro_interaction_cols or macro_available or DEFAULT_MACRO_INTERACTION_COLS
    interaction_macro_available = [c for c in interaction_base_cols if c in df.columns]
    if include_macro_sic_interactions and interaction_macro_available:
        interaction_df, interaction_cols = _build_macro_sic_interactions(
            df, interaction_macro_available, sic2_col="sic2"
        )
        df = pd.concat([df, interaction_df], axis=1)

    lo = df.groupby("month_end")["ret_eom_t+1"].transform(lambda x: x.quantile(0.01))
    hi = df.groupby("month_end")["ret_eom_t+1"].transform(lambda x: x.quantile(0.99))
    df["ret_eom_t+1"] = df["ret_eom_t+1"].clip(lower=lo, upper=hi)

    lo = df.groupby("month_end")["ret_eom_t+1_excess"].transform(lambda x: x.quantile(0.01))
    hi = df.groupby("month_end")["ret_eom_t+1_excess"].transform(lambda x: x.quantile(0.99))
    df["ret_eom_t+1_excess"] = df["ret_eom_t+1_excess"].clip(lower=lo, upper=hi)

    gkx_cols = [c for c in GKX_94 if c in df.columns]
    feature_cols = _dedupe_keep_order(
        gkx_cols + [c for c in macro_feature_cols if c in df.columns] + sic_dummy_cols + interaction_cols
    )
    base_cols = ["ticker", "month_end", "sic2", "spy_ret_eom_t+1", "ret_eom_t+1", "ret_eom_t+1_excess"]
    model_cols = [c for c in base_cols + feature_cols if c in df.columns]

    model_df = df[model_cols].copy()
    model_df = model_df[model_df["ret_eom_t+1"].notna()].reset_index(drop=True)
    model_df = model_df.replace([np.inf, -np.inf], np.nan)

    etf_flags = _load_etf_flags(universe_file)
    if etf_flags is not None:
        model_df = model_df.merge(etf_flags, on="ticker", how="left")
        model_df = model_df[(model_df["is_etf"] == False) | (model_df["ticker"] == "SPY")]
        model_df = model_df.drop(columns=["is_etf"])

    return model_df


def _parse_csv_list(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [c.strip() for c in raw.split(",") if c.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build notebook-style monthly model_df base artifact")
    parser.add_argument("--run_tag", type=str, required=True, help="Run tag under data/processed")
    parser.add_argument("--processed_root", type=str, default="data/processed", help="Processed root directory")
    parser.add_argument("--ff_factors_path", type=str, default="data/downloaded/F-F_Research_Data_Factors.csv")
    parser.add_argument("--universe_file", type=str, default=None, help="Universe CSV with is_etf for ETF filtering")
    parser.add_argument("--macro_cols", type=str, default=None, help="Comma-separated macro columns override")
    parser.add_argument(
        "--macro_interaction_cols",
        type=str,
        default=None,
        help="Comma-separated macro columns used for macro x SIC interactions",
    )
    parser.add_argument(
        "--no_macro_sic_interactions",
        action="store_true",
        help="Disable macro x SIC interaction feature generation",
    )
    parser.add_argument("--output", type=str, default=None, help="Override output parquet path")
    args = parser.parse_args()

    processed_dir = os.path.join(args.processed_root, args.run_tag)
    gkx_path = os.path.join(processed_dir, "gkx_monthly.parquet")
    daily_path = os.path.join(processed_dir, "final_dataset.parquet")
    output_path = args.output or os.path.join(processed_dir, "model_df_base.parquet")

    gkx = pd.read_parquet(gkx_path)
    daily = pd.read_parquet(daily_path)
    model_df = build_model_df(
        gkx,
        daily,
        ff_factors_path=args.ff_factors_path,
        universe_file=args.universe_file,
        macro_cols=_parse_csv_list(args.macro_cols),
        macro_interaction_cols=_parse_csv_list(args.macro_interaction_cols),
        include_macro_sic_interactions=not args.no_macro_sic_interactions,
    )
    model_df.to_parquet(output_path, index=False)
    print(f"Saved model_df base to: {output_path}")
    print(f"Shape: {model_df.shape}")
