#!/usr/bin/env python3
"""
FINRA OTC (ATS & Non-ATS) Weekly Summary — ATS share + WoW + 8-week trend
- Pulls latest N weeks from FINRA Query API (weeklySummary).
- Computes symbol-level ATS vs non-ATS (off-exchange only).
- Outputs CSVs, two quick bar charts, and a multi-week trend chart.

Run:
  python src/finra_ats_volume_trends.py --weeks 8 --tiers T1 T2
"""

from __future__ import annotations
import os
import sys
from typing import List, Tuple, Dict
from datetime import datetime
import requests
import pandas as pd
from dateutil import parser as dtparser

FINRA_BASE = "https://api.finra.org"
PARTITIONS_URL = f"{FINRA_BASE}/partitions/group/otcmarket/name/weeklysummary"
DATA_URL = f"{FINRA_BASE}/data/group/OTCMarket/name/weeklysummary"
HEADERS_JSON = {"Accept": "application/json"}

# Symbol-level summaryTypeCodes for off-exchange
ATS_CODE = "ATS_W_SMBL"
OTC_CODE = "OTC_W_SMBL"  # non-ATS off-exchange

DEFAULT_WEEKS = 8
DEFAULT_TIERS = ["T1", "T2"]  # NMS; use ["OTC"] or include it for OTCE

def _ensure_dirs():
    os.makedirs("output", exist_ok=True)
    os.makedirs("charts", exist_ok=True)

def fetch_available_weeks() -> List[str]:
    """Return available weekStartDate strings (yyyy-mm-dd), newest first."""
    r = requests.get(PARTITIONS_URL, headers=HEADERS_JSON, timeout=30)
    r.raise_for_status()
    payload = r.json()

    def is_date(s: str) -> bool:
        try:
            dtparser.isoparse(s); return True
        except Exception:
            return False

    weeks = set()
    for entry in payload.get("availablePartitions", []):
        for token in entry.get("partitions", []):
            if isinstance(token, str) and is_date(token):
                weeks.add(token)

    weeks_sorted = sorted(weeks, key=lambda d: dtparser.isoparse(d), reverse=True)
    return weeks_sorted

def post_weekly_summary(week: str,
                        tiers: List[str],
                        summary_codes: List[str],
                        fields: List[str],
                        limit: int = 500_000) -> pd.DataFrame:
    """POST query for one week across tiers and summary codes; return DataFrame."""
    payload = {
        "fields": fields,
        "limit": limit,
        "compareFilters": [{"fieldName": "weekStartDate", "fieldValue": week, "compareType": "EQUAL"}],
        "domainFilters": [
            {"fieldName": "tierIdentifier", "values": tiers},
            {"fieldName": "summaryTypeCode", "values": summary_codes},
        ],
    }
    r = requests.post(DATA_URL, headers=HEADERS_JSON, json=payload, timeout=90)
    if r.status_code == 401:
        raise RuntimeError("FINRA API returned 401 for weeklySummary. Try again later.")
    r.raise_for_status()
    return pd.DataFrame(r.json())

def build_week_df(week: str, tiers: List[str]) -> pd.DataFrame:
    """Get ATS vs non-ATS rows for a week and pivot to columns."""
    fields = [
        "issueSymbolIdentifier","issueName","weekStartDate",
        "summaryTypeCode","totalWeeklyShareQuantity"
    ]
    df = post_weekly_summary(week, tiers, [ATS_CODE, OTC_CODE], fields)
    if df.empty:
        return df

    keep = ["issueSymbolIdentifier","issueName","weekStartDate","summaryTypeCode","totalWeeklyShareQuantity"]
    df = df[keep].copy()
    df["issueSymbolIdentifier"] = df["issueSymbolIdentifier"].astype(str).str.upper()


    pivot = (df.pivot_table(
                index=["issueSymbolIdentifier","issueName","weekStartDate"],
                columns="summaryTypeCode",
                values="totalWeeklyShareQuantity",
                aggfunc="sum",
                fill_value=0)
             .reset_index())

    if ATS_CODE not in pivot.columns: pivot[ATS_CODE] = 0
    if OTC_CODE not in pivot.columns: pivot[OTC_CODE] = 0

    pivot["total_off_exchange_shares"] = pivot[ATS_CODE] + pivot[OTC_CODE]
    pivot["ats_share_pct"] = pivot.apply(
        lambda r: (r[ATS_CODE] / r["total_off_exchange_shares"]) * 100 if r["total_off_exchange_shares"] > 0 else 0.0,
        axis=1
    )
    return pivot.sort_values("total_off_exchange_shares", ascending=False)

def compute_wow(current: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Merge two weekly frames by symbol and compute WoW deltas."""
    if current.empty or prior.empty:
        return pd.DataFrame()

    c = current.rename(columns={
        ATS_CODE: "ats_shares_cur",
        OTC_CODE: "otc_shares_cur",
        "ats_share_pct": "ats_share_pct_cur",
        "total_off_exchange_shares": "total_shares_cur"
    })
    p = prior.rename(columns={
        ATS_CODE: "ats_shares_prev",
        OTC_CODE: "otc_shares_prev",
        "ats_share_pct": "ats_share_pct_prev",
        "total_off_exchange_shares": "total_shares_prev"
    })
    on_cols = ["issueSymbolIdentifier"]
    merged = c.merge(
        p[on_cols + ["ats_shares_prev","otc_shares_prev","ats_share_pct_prev","total_shares_prev"]],
        on=on_cols, how="left"
    )

    merged["ats_share_pct_delta"] = merged["ats_share_pct_cur"] - merged["ats_share_pct_prev"]
    def _pct(a, b): 
        return ((a-b)/b*100) if (b is not None and b and b>0) else None
    merged["total_shares_wow_pct"] = merged.apply(lambda r: _pct(r["total_shares_cur"], r.get("total_shares_prev")), axis=1)
    return merged

def make_charts(df_latest: pd.DataFrame, outdir="charts", top_n: int = 15) -> Tuple[str, str]:
    """Bar charts: top ATS-share; top WoW movers (latest vs prior)."""
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    chart1 = os.path.join(outdir, "ats_share_top_symbols_latest.png")
    chart2 = os.path.join(outdir, "ats_share_wow_delta_top_symbols.png")

    # Top by ATS-share % with a small liquidity floor if available
    latest = df_latest.copy()
    if "total_shares_cur" in latest.columns:
        latest = latest[latest["total_shares_cur"] > 100_000]
        ycol = "ats_share_pct_cur"
    else:
        ycol = "ats_share_pct"
    latest = latest.sort_values(ycol, ascending=False).head(top_n)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 6))
    plt.bar(latest["issueSymbolIdentifier"], latest[ycol])
    plt.title("Top Symbols by ATS Share % (Latest Week)")
    plt.ylabel("ATS Share (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(chart1, dpi=144)
    plt.close()

    # WoW movers (if deltas exist)
    if "ats_share_pct_delta" in df_latest.columns and "total_shares_cur" in df_latest.columns:
        movers = df_latest[df_latest["total_shares_cur"] > 100_000].copy()
        movers = movers[movers["ats_share_pct_delta"].notna()]
        movers = movers.sort_values("ats_share_pct_delta", ascending=False).head(top_n)
        if not movers.empty:
            plt.figure(figsize=(11, 6))
            plt.bar(movers["issueSymbolIdentifier"], movers["ats_share_pct_delta"])
            plt.title("Top WoW Movers in ATS Share % (Latest vs Prior)")
            plt.ylabel("Δ ATS Share (pp)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(chart2, dpi=144)
            plt.close()
        else:
            chart2 = ""
    else:
        chart2 = ""

    return chart1, chart2

def make_trend_chart(week_frames: Dict[str, pd.DataFrame],
                     outdir: str = "charts",
                     top_n: int = 10) -> str:
    """
    Line chart: ATS share % over the last N weeks for the top N tickers by
    cumulative off-exchange volume across those weeks.
    """
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "ats_share_trend_top10.png")

    if not week_frames:
        return ""

    # Stack weeks with a common schema
    rows = []
    for wk, df in week_frames.items():
        if df.empty: 
            continue
        tmp = df.rename(columns={
            ATS_CODE: "ats_shares",
            OTC_CODE: "otc_shares",
            "total_off_exchange_shares": "total_shares"
        }).copy()
        tmp["week"] = wk
        rows.append(tmp[["issueSymbolIdentifier","issueName","week","ats_share_pct","total_shares"]])
    data = pd.concat(rows, ignore_index=True)
    if data.empty:
        return ""

    # Top N by cumulative total_shares
    totals = (data.groupby(["issueSymbolIdentifier","issueName"])["total_shares"]
                  .sum().sort_values(ascending=False).head(top_n))
    keep = totals.reset_index()[["issueSymbolIdentifier"]]
    data = data.merge(keep, on="issueSymbolIdentifier", how="inner")

    # Pivot weeks → rows, symbols → columns (values = ats_share_pct)
    # Sort weeks ascending on the x-axis
    data["week_dt"] = data["week"].apply(dtparser.isoparse)
    trend = (data.pivot_table(index="week_dt", columns="issueSymbolIdentifier",
                              values="ats_share_pct", aggfunc="mean")
                  .sort_index())

    # Plot
    plt.figure(figsize=(12, 7))
    for col in trend.columns:
        plt.plot(trend.index, trend[col], label=col)
    plt.title("ATS Share % — 8-Week Trend (Top 10 by Off-Exchange Volume)")
    plt.ylabel("ATS Share (%)")
    plt.xlabel("Week")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend(ncol=2, fontsize=9)
    plt.savefig(outpath, dpi=144)
    plt.close()
    return outpath

def main(weeks: int = DEFAULT_WEEKS, tiers: List[str] = None):
    _ensure_dirs()
    tiers = tiers or DEFAULT_TIERS

    # Fetch most recent N weeks from FINRA
    available = fetch_available_weeks()
    if not available:
        print("No weeks available from FINRA.")
        sys.exit(1)
    use_weeks = available[:max(weeks, 2)]
    print(f"Weeks pulled: {use_weeks}")

    # Build per-week frames → store and write raw CSV
    week_frames: Dict[str, pd.DataFrame] = {}
    for w in use_weeks:
        dfw = build_week_df(w, tiers)
        dfw = dfw.rename(columns={
            ATS_CODE: "ats_shares",
            OTC_CODE: "otc_shares",
            "total_off_exchange_shares": "total_shares",
            "ats_share_pct": "ats_share_pct"
        })
        week_frames[w] = dfw
        dfw.to_csv(f"output/weekly_{w}.csv", index=False)

    # Latest vs prior WoW
    latest_week = use_weeks[0]
    latest_df = week_frames[latest_week].copy()
    if len(use_weeks) > 1:
        prior_week = use_weeks[1]
        prior_df = week_frames[prior_week].copy()
        wow = compute_wow(
            current=latest_df.rename(columns={"ats_shares":"ATS_W_SMBL",
                                              "otc_shares":"OTC_W_SMBL",
                                              "total_shares":"total_off_exchange_shares",
                                              "ats_share_pct":"ats_share_pct"}),
            prior=prior_df.rename(columns={"ats_shares":"ATS_W_SMBL",
                                           "otc_shares":"OTC_W_SMBL",
                                           "total_shares":"total_off_exchange_shares",
                                           "ats_share_pct":"ats_share_pct"})
        )
        if not wow.empty:
            wow.to_csv("output/latest_vs_prior_wow.csv", index=False)
            latest_df = wow  # reuse for bar charts

    # Bar charts (latest snapshot)
    c1, c2 = make_charts(latest_df)
    print(f"Wrote: {c1}{' and ' + c2 if c2 else ''}")

    # New multi-week trend chart
    trend_path = make_trend_chart(week_frames, top_n=10)
    if trend_path:
        print(f"Wrote: {trend_path}")

    print("Done. See ./output and ./charts.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="FINRA OTC ATS vs non-ATS trends")
    p.add_argument("--weeks", type=int, default=DEFAULT_WEEKS, help="Recent weeks to pull (>=2).")
    p.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS, help="T1 T2 (NMS) by default; add OTC if needed.")
    args = p.parse_args()
    main(weeks=args.weeks, tiers=args.tiers)
