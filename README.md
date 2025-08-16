# FINRA ATS (includes dark-pool venues) & Off-Exchange Volume Trends

**What this does**  
Uses FINRA’s public **off-exchange** weekly dataset to show, by symbol:
- How much traded on **ATS venues (includes dark pools)** vs **non-ATS off-exchange** this week.
- The **week-over-week change** in ATS share (percentage points).
- An **8-week trend** of ATS share for the 10 most active tickers.

**Why care**  
A lot of U.S. equity trading occurs **off the exchanges**. Seeing how much of that flow is on **ATS venues** (which include dark-pool venues) — and how it **shifts week to week** — is useful for execution/routing context and market-structure awareness. This project uses **only public data** from FINRA.

> **Terminology:** ATS = Alternative Trading System. Many ATS venues are what people call *dark pools*, but **not all ATS are dark pools**. This project uses FINRA’s ATS vs non-ATS categories; it does not attempt to split “dark” vs “non-dark” within ATS.

---

## How to run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python finra_ats_offexchange_trends.py --weeks 8 --tiers T1 T2
