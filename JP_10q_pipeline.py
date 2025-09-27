# --- prerequisites ---
# pip install requests pandas openpyxl python-dateutil

import requests
import pandas as pd
from dateutil import parser as dateparser
from pathlib import Path

CIK = "0000019617"  # JP Morgan, Inc.
USER_AGENT = "Thai Findeep/1.0 thaiviet0703@gmail.com"
OUTFILE = Path("jp_morgan_10Q.xlsx")

US_GAAP_TAGS = [
    "Revenues",
    "CostOfRevenue",
    "GrossProfit",
    "OperatingExpenses",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities",
]

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})

def get_json(url, timeout=60):
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_submissions(cik: str) -> dict:
    return get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")

def get_companyfacts(cik: str) -> dict:
    return get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")

# 1) Latest Tesla 10-Q metadata (filing dates/periods/accessions)

from datetime import datetime, timedelta

subs = get_submissions(CIK)
recent = subs.get("filings", {}).get("recent", {})
forms = recent.get("form", [])
filing_dates = recent.get("filingDate", [])
periods = recent.get("reportDate", [])
accessions = recent.get("accessionNumber", [])

# Calculate the date 15 years ago from today
fifteen_years_ago = (datetime.now() - timedelta(days=365*15)).date()

fifteen_q_meta = []
for form, fdate, period, acc in zip(forms, filing_dates, periods, accessions):
    if form == "10-Q":
        # Parse the filing date
        try:
            filing_dt = pd.to_datetime(fdate).date()
        except Exception:
            continue
        if filing_dt >= fifteen_years_ago:
            fifteen_q_meta.append({
                "filingDate": fdate,
                "reportPeriod": period or None,
                "accession": acc
            })

fifteen_q_meta = sorted(fifteen_q_meta, key=lambda x: x["filingDate"], reverse=True)
allowed_accns = set(x["accession"].replace("-", "") for x in fifteen_q_meta if x["accession"])

# 2) All JP Morgan XBRL facts
facts = get_companyfacts(CIK)

def extract_quarterly_values(facts_json, tag):
    """
    Return list of dicts from us-gaap/{tag} for 10-Q entries.
    Keep USD for $ amounts; keep per-share for EPS.
    """
    out = []
    tag_obj = facts_json.get("facts", {}).get("us-gaap", {}).get(tag)
    if not tag_obj:
        return out

    for unit, arr in tag_obj.get("units", {}).items():
        for row in arr:
            form = row.get("form")
            end = row.get("end")
            val = row.get("val")
            accn = row.get("accn")
            if form != "10-Q" or end is None or val is None or accn is None:
                continue

            # units filtering
            if tag == "EarningsPerShareDiluted":
                # EPS comes as per-share; allow any 'per share' unit variant
                if "share" not in unit.lower():
                    continue
            else:
                if unit != "USD":
                    continue

            out.append({
                "tag": tag,
                "form": form,
                "end": end,
                "value": val,
                "uom": unit,
                "accn": accn
            })
    return out

rows = []
for tag in US_GAAP_TAGS:
    rows.extend(extract_quarterly_values(facts, tag))

raw_df = pd.DataFrame(rows)
if raw_df.empty:
    raise SystemExit("No quarterly values found. Check your User-Agent or tags.")

# 3) Keep only entries tied to our latest known 10-Q accessions
raw_df["accn_stripped"] = raw_df["accn"].str.replace("-", "", regex=False)
raw_df = raw_df[raw_df["accn_stripped"].isin(allowed_accns)].copy()


# 4) Build Summary sheet (all available quarters from last 15 years)
raw_df["end_dt"] = pd.to_datetime(raw_df["end"])
all_quarters = raw_df[["end_dt"]].drop_duplicates().sort_values("end_dt", ascending=False)["end_dt"].tolist()

summary_df = (
    raw_df[raw_df["end_dt"].isin(all_quarters)]
    .assign(Quarter=lambda d: d["end_dt"].dt.strftime("%Y-%m-%d"))
    .pivot_table(index="tag", columns="Quarter", values="value", aggfunc="last")
    .sort_index()
)

summary_df = summary_df[sorted(summary_df.columns, reverse=True)]

# 5) Write Excel

# Delete old Excel file if it exists
if OUTFILE.exists():
    OUTFILE.unlink()

with pd.ExcelWriter(OUTFILE, engine="openpyxl") as xls:
    summary_df.to_excel(xls, sheet_name="Summary")
    (raw_df
        .drop(columns=["accn_stripped", "end_dt"])
        .sort_values(["tag","end"])
        .to_excel(xls, sheet_name="RawFacts", index=False))

print(f"Done. Wrote {OUTFILE.resolve()}")
