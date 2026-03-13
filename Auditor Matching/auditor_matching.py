#!/usr/bin/env python3
"""
Auditor Matching Script
=======================
Matches engagement partners from accounting/audit firms with political
donation contribution records based on:
  a) Name  – first + last name, with middle name for additional specificity
  b) Company – intensive standardization to reconcile variant firm names
  c) Occupation – restricted to audit- and legal-field keywords

Scoring tiers (cutoff >= 90):
  Tier 1 (highest accuracy)  : a + b + c
  Tier 2 (second highest)    : a + b      (when occupation is unavailable)
  Tier 3 (third highest)     : a + c      (when company is unavailable)

Inputs
------
Engagement partners:
  <BASE_DIR>/Engagement Partner Firm and ID Sample Unmatched Ideology Pass 2.csv

Contribution records:
  <BASE_DIR>/Raw Itemized Contribution Records/contribDB_<YEAR>.csv
  for years: 1980, 1982, 1984 … 2014, 2016, 2018, 2020, 2022, 2024

Outputs
-------
  <BASE_DIR>/Auditor Matching/Matching Output/  – matched results CSV
  <BASE_DIR>/Auditor Matching/Log/              – run log file

Usage
-----
  python auditor_matching.py [--base-dir "E:\\Political Donations Files"]

Dependencies
------------
  pip install pandas rapidfuzz
"""

import argparse
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Relative path anchors (resolved at runtime from --base-dir)
PARTNER_FILE = "Engagement Partner Firm and ID Sample Unmatched Ideology Pass 2.csv"
CONTRIB_SUBDIR = "Raw Itemized Contribution Records"
OUTPUT_SUBDIR = "Auditor Matching/Matching Output"
LOG_SUBDIR = "Auditor Matching/Log"

CONTRIB_YEARS = list(range(1980, 2015, 2)) + [2016, 2018, 2020, 2022, 2024]

# Score cutoff for a match to be kept
SCORE_CUTOFF = 90.0

# Minimum name score to bother scoring company/occupation.
# Derived from the best-case composite for each tier:
#   Tier1: 0.50*n + 0.35*100 + 0.15*100 >= 90  →  n >= 80
#   Tier2: 0.60*n + 0.40*100             >= 90  →  n >= 83.3
#   Tier3: 0.70*n + 0.30*100             >= 90  →  n >= 85.7
# We use the Tier1 minimum (most lenient) minus a small safety margin.
NAME_PREFILTER_THRESHOLD = max(0.0, (SCORE_CUTOFF - 50.0) / 0.50 - 2.0)  # 78.0

# ---------------------------------------------------------------------------
# Audit / Legal occupation keyword lists
# ---------------------------------------------------------------------------

AUDIT_OCCUPATION_KEYWORDS = [
    "audit", "auditor", "assurance", "cpa", "certified public accountant",
    "public accountant", "accountant", "accounting", "controller",
    "comptroller", "forensic accountant", "internal audit",
    "external audit", "tax", "taxation", "tax advisor", "tax consultant",
    "tax manager", "tax partner", "tax director", "tax professional",
    "engagement partner", "managing partner", "audit partner",
    "senior partner", "partner", "principal", "director of audit",
    "attest", "review engagement",
]

LEGAL_OCCUPATION_KEYWORDS = [
    "attorney", "lawyer", "counsel", "legal", "solicitor", "barrister",
    "paralegal", "law clerk", "associate attorney", "partner attorney",
    "general counsel", "deputy counsel", "managing counsel",
    "corporate counsel", "compliance", "regulatory", "judge",
    "magistrate", "public defender", "prosecutor",
]

ALL_OCC_KEYWORDS = set(
    kw.lower() for kw in AUDIT_OCCUPATION_KEYWORDS + LEGAL_OCCUPATION_KEYWORDS
)

# ---------------------------------------------------------------------------
# Common company-name tokens to strip / normalize
# ---------------------------------------------------------------------------

COMPANY_SUFFIX_RE = re.compile(
    r"\b("
    r"llp|lllp|llc|lc|plc|pllc|pc|pa|na|inc|incorporated|corp|corporation|"
    r"co|company|companies|ltd|limited|group|grp|assoc|associates|"
    r"associate|services|service|solutions|consulting|consultants|"
    r"management|mgmt|international|intl|national|natl|"
    r"certified public accountants|cpas|cpa|"
    r"chartered accountants|cas|ca"
    r")\b",
    re.IGNORECASE,
)

AMP_RE = re.compile(r"\s*&\s*")          # & → and
PUNCT_RE = re.compile(r"[^\w\s]")        # strip non-word, non-space
WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _to_ascii(text: str) -> str:
    """Decompose unicode characters and drop combining marks."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def normalize_name(raw: str) -> dict:
    """
    Parse a raw contributor name string into first, middle, last tokens.

    Handles common formats:
      "LAST, FIRST MIDDLE"
      "FIRST MIDDLE LAST"
      "FIRST LAST"

    Returns dict with keys: first, middle, last, full
    """
    if not isinstance(raw, str) or not raw.strip():
        return {"first": "", "middle": "", "last": "", "full": ""}

    text = _to_ascii(raw).upper().strip()
    # Strip common suffixes
    text = re.sub(
        r"\b(JR|SR|II|III|IV|V|ESQ|PHD|MD|CPA|CFA|MBA)\b\.?",
        "",
        text,
    )
    text = WHITESPACE_RE.sub(" ", text).strip()

    # Detect "LAST, FIRST MIDDLE" format
    if "," in text:
        parts = text.split(",", 1)
        last = parts[0].strip()
        rest = parts[1].strip().split()
        first = rest[0] if rest else ""
        middle = " ".join(rest[1:]) if len(rest) > 1 else ""
    else:
        tokens = text.split()
        if len(tokens) == 1:
            first, middle, last = tokens[0], "", ""
        elif len(tokens) == 2:
            first, middle, last = tokens[0], "", tokens[1]
        else:
            first = tokens[0]
            last = tokens[-1]
            middle = " ".join(tokens[1:-1])

    full = f"{first} {last}".strip()
    return {"first": first, "middle": middle, "last": last, "full": full}


def normalize_company(raw: str) -> str:
    """
    Aggressively standardize company/employer name for fuzzy comparison.
    Steps: ascii-fold → lower → & → and → strip punctuation →
           remove common suffixes → collapse whitespace.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    text = _to_ascii(raw).lower()
    text = AMP_RE.sub(" and ", text)
    text = PUNCT_RE.sub(" ", text)
    text = COMPANY_SUFFIX_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def occupation_in_scope(raw: str) -> bool:
    """Return True if the occupation string contains an audit/legal keyword."""
    if not isinstance(raw, str) or not raw.strip():
        return False
    text = raw.lower()
    return any(kw in text for kw in ALL_OCC_KEYWORDS)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def name_score(name_a: dict, name_b: dict) -> float:
    """
    Compute a 0-100 name similarity score.
    Primary: full "FIRST LAST" token-sort ratio.
    Bonus: if both have a middle name, add a weighted boost (up to +5 pts).
    """
    if not name_a["full"] or not name_b["full"]:
        return 0.0

    base = fuzz.token_sort_ratio(name_a["full"], name_b["full"])

    # Middle-name bonus: boosts score for very close matches
    if name_a["middle"] and name_b["middle"]:
        mid_sim = fuzz.ratio(name_a["middle"], name_b["middle"])
        bonus = 5.0 * (mid_sim / 100.0)
        base = min(100.0, base + bonus)

    return float(base)


def company_score(comp_a: str, comp_b: str) -> float:
    """Token-set ratio on standardised company strings (0-100)."""
    if not comp_a or not comp_b:
        return 0.0
    return float(fuzz.token_set_ratio(comp_a, comp_b))


def compute_tier_score(
    n_score: float,
    c_score: float,
    o_match: bool,
    has_company: bool,
    has_occ: bool,
) -> tuple[float, str]:
    """
    Return (composite_score, tier_label) based on data availability.

    Tier 1 (a+b+c): name 50%, company 35%, occupation 15%
    Tier 2 (a+b):   name 60%, company 40%
    Tier 3 (a+c):   name 70%, occupation 30%
    """
    o_score = 100.0 if o_match else 0.0

    if has_company and has_occ:
        score = 0.50 * n_score + 0.35 * c_score + 0.15 * o_score
        tier = "Tier1_Name_Company_Occupation"
    elif has_company:
        score = 0.60 * n_score + 0.40 * c_score
        tier = "Tier2_Name_Company"
    elif has_occ:
        score = 0.70 * n_score + 0.30 * o_score
        tier = "Tier3_Name_Occupation"
    else:
        # Name only – not in scoring matrix; use raw name score
        score = n_score
        tier = "NameOnly"

    return round(score, 2), tier


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_partners(path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading engagement partners from: %s", path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    logger.info("  Loaded %d rows, columns: %s", len(df), list(df.columns))
    return df


def load_contributions(contrib_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    frames = []
    for year in CONTRIB_YEARS:
        fpath = contrib_dir / f"contribDB_{year}.csv"
        if not fpath.exists():
            logger.warning("  Contribution file not found, skipping: %s", fpath)
            continue
        logger.info("  Loading contributions for year %d …", year)
        df = pd.read_csv(fpath, dtype=str, low_memory=False)
        df["_source_year"] = str(year)
        frames.append(df)
        logger.info("    %d rows loaded.", len(df))

    if not frames:
        logger.error("No contribution files found in %s", contrib_dir)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total contribution records loaded: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Main matching routine
# ---------------------------------------------------------------------------

def run_matching(partners: pd.DataFrame, contribs: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Iterate over each engagement partner and score against all contribution
    records whose name passes a cheap pre-filter (last-name token match).
    Returns a DataFrame of matches at or above SCORE_CUTOFF.
    """

    # ---- Detect partner columns ----------------------------------------
    p_name_col = find_col(partners, ["name", "partner_name", "engagementpartner",
                                      "engagement_partner", "fullname", "full_name"])
    p_firm_col = find_col(partners, ["firm", "firm_name", "firmname", "company",
                                      "employer", "organization"])
    p_occ_col  = find_col(partners, ["occupation", "occ", "title", "position",
                                      "job_title", "jobtitle"])

    logger.info("Partner columns  → name: %s | firm: %s | occupation: %s",
                p_name_col, p_firm_col, p_occ_col)

    if p_name_col is None:
        logger.error("Cannot identify a name column in the engagement partner file. "
                     "Please check column names.")
        return pd.DataFrame()

    # ---- Detect contribution columns ------------------------------------
    c_name_col = find_col(contribs, ["name", "contributor_name", "contribname",
                                      "fullname", "full_name", "donor_name",
                                      "donorname"])
    c_emp_col  = find_col(contribs, ["employer", "org", "organization", "company",
                                      "contrib_employer", "employer_name"])
    c_occ_col  = find_col(contribs, ["occupation", "occ", "contrib_occupation"])

    logger.info("Contribution columns → name: %s | employer: %s | occupation: %s",
                c_name_col, c_emp_col, c_occ_col)

    if c_name_col is None:
        logger.error("Cannot identify a name column in the contribution records. "
                     "Please check column names.")
        return pd.DataFrame()

    # ---- Pre-process contribution records --------------------------------
    logger.info("Pre-processing contribution records …")
    contribs["_norm_name"]    = contribs[c_name_col].apply(normalize_name)
    contribs["_norm_company"] = (
        contribs[c_emp_col].apply(normalize_company)
        if c_emp_col
        else pd.Series([""] * len(contribs), index=contribs.index, dtype=str)
    )
    contribs["_occ_match"]    = (
        contribs[c_occ_col].apply(occupation_in_scope)
        if c_occ_col
        else pd.Series([False] * len(contribs), index=contribs.index, dtype=bool)
    )
    contribs["_last_token"]   = contribs["_norm_name"].apply(
        lambda d: d["last"].upper() if d else ""
    )

    # Build last-name index for fast pre-filtering (vectorized via groupby)
    last_name_index: dict[str, list[int]] = {
        key: list(group.index)
        for key, group in contribs.groupby("_last_token")
        if key  # exclude empty-string bucket
    }

    # ---- Match loop -------------------------------------------------------
    results = []
    total_partners = len(partners)

    # Pre-build a positional lookup for contribution helper columns
    # (avoids repeated .loc[] calls in the hot inner loop)
    contrib_col_data = {
        "_norm_name":    contribs["_norm_name"].to_dict(),
        "_norm_company": contribs["_norm_company"].to_dict(),
        "_occ_match":    contribs["_occ_match"].to_dict(),
        "_source_year":  contribs["_source_year"].to_dict(),
    }
    if c_name_col:
        contrib_col_data[c_name_col] = contribs[c_name_col].to_dict()
    if c_emp_col:
        contrib_col_data[c_emp_col] = contribs[c_emp_col].to_dict()
    if c_occ_col:
        contrib_col_data[c_occ_col] = contribs[c_occ_col].to_dict()
    # Snapshot all original contrib columns for result assembly
    orig_contrib_cols = [col for col in contribs.columns if not col.startswith("_")]
    contrib_orig_data = {col: contribs[col].to_dict() for col in orig_contrib_cols}

    def _safe_str(row: pd.Series, col: str | None) -> str:
        """Return a non-null string value from row[col], or '' if col is None/NaN."""
        if col is None:
            return ""
        val = row.get(col)
        return str(val) if pd.notna(val) else ""

    for enum_idx, (p_idx, p_row) in enumerate(partners.iterrows()):
        raw_p_name = _safe_str(p_row, p_name_col)
        p_name_d   = normalize_name(raw_p_name)

        raw_p_firm = _safe_str(p_row, p_firm_col)
        p_firm_n   = normalize_company(raw_p_firm)

        raw_p_occ  = _safe_str(p_row, p_occ_col)
        p_occ_ok   = occupation_in_scope(raw_p_occ)

        if enum_idx % 100 == 0:
            logger.info("  Processing partner %d / %d : %s",
                        enum_idx + 1, total_partners, raw_p_name)

        last_key = p_name_d["last"].upper()
        if not last_key:
            continue  # Skip partners with no parseable last name

        # Pre-filter: only consider contribution rows with matching last-name token
        candidate_idxs = last_name_index.get(last_key, [])
        if not candidate_idxs:
            continue

        for c_idx in candidate_idxs:
            c_name_d   = contrib_col_data["_norm_name"][c_idx]
            c_firm_n   = contrib_col_data["_norm_company"][c_idx]
            c_occ_ok   = bool(contrib_col_data["_occ_match"][c_idx])

            # --- Name score (fast pre-filter before heavier company/occ scoring)
            n_sc = name_score(p_name_d, c_name_d)
            if n_sc < NAME_PREFILTER_THRESHOLD:
                continue

            # --- Company availability
            has_company = bool(p_firm_n and c_firm_n)
            c_sc = company_score(p_firm_n, c_firm_n) if has_company else 0.0

            # --- Occupation availability (both sides must be in-scope)
            has_occ = p_occ_ok and c_occ_ok

            composite, tier = compute_tier_score(n_sc, c_sc, has_occ,
                                                  has_company, has_occ)

            if composite < SCORE_CUTOFF:
                continue

            # Assemble result row
            record = {
                "partner_row_id":        p_idx,
                "partner_name_raw":      raw_p_name,
                "partner_firm_raw":      raw_p_firm,
                "partner_occupation":    raw_p_occ,
                "contrib_row_id":        c_idx,
                "contrib_name_raw":      contrib_col_data.get(c_name_col, {}).get(c_idx, ""),
                "contrib_employer_raw":  contrib_col_data.get(c_emp_col, {}).get(c_idx, "") if c_emp_col else "",
                "contrib_occupation":    contrib_col_data.get(c_occ_col, {}).get(c_idx, "") if c_occ_col else "",
                "contrib_source_year":   contrib_col_data["_source_year"][c_idx],
                "score_name":            n_sc,
                "score_company":         c_sc,
                "score_occ_match":       has_occ,
                "composite_score":       composite,
                "match_tier":            tier,
            }
            # Include all original partner columns
            for col in partners.columns:
                record[f"p_{col}"] = p_row[col]
            # Include all original contribution columns (excluding helper cols)
            for col in orig_contrib_cols:
                record[f"c_{col}"] = contrib_orig_data[col][c_idx]

            results.append(record)

    logger.info("Total matches found (score >= %s): %d", SCORE_CUTOFF, len(results))
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"auditor_matching_{timestamp}.log"

    logger = logging.getLogger("auditor_matching")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Auditor Engagement Partner Matching")
    parser.add_argument(
        "--base-dir",
        default=r"E:\Political Donations Files",
        help="Root directory containing all input subfolders.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)

    # Resolve paths
    log_dir     = base / LOG_SUBDIR
    output_dir  = base / OUTPUT_SUBDIR
    partner_csv = base / PARTNER_FILE
    contrib_dir = base / CONTRIB_SUBDIR

    # Setup logging first
    logger = setup_logging(log_dir)

    logger.info("=" * 70)
    logger.info("Auditor Matching Run")
    logger.info("Base directory   : %s", base)
    logger.info("Partner file     : %s", partner_csv)
    logger.info("Contributions dir: %s", contrib_dir)
    logger.info("Output dir       : %s", output_dir)
    logger.info("Score cutoff     : %s", SCORE_CUTOFF)
    logger.info("=" * 70)

    # Validate inputs
    if not partner_csv.exists():
        logger.error("Engagement partner file not found: %s", partner_csv)
        sys.exit(1)
    if not contrib_dir.exists():
        logger.error("Contribution records directory not found: %s", contrib_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    partners = load_partners(partner_csv, logger)
    contribs = load_contributions(contrib_dir, logger)

    if contribs.empty:
        logger.error("No contribution records loaded. Exiting.")
        sys.exit(1)

    # Run matching
    logger.info("Starting matching …")
    t_start = datetime.now()
    matches = run_matching(partners, contribs, logger)
    elapsed = (datetime.now() - t_start).total_seconds()
    logger.info("Matching completed in %.1f seconds.", elapsed)

    # Save output
    if not matches.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = output_dir / f"auditor_matches_{timestamp}.csv"
        matches.to_csv(out_file, index=False)
        logger.info("Matches saved to: %s", out_file)

        # Also save a summary by tier
        summary = (
            matches.groupby("match_tier")
            .agg(count=("composite_score", "count"),
                 mean_score=("composite_score", "mean"),
                 min_score=("composite_score", "min"),
                 max_score=("composite_score", "max"))
            .reset_index()
        )
        summary_file = output_dir / f"auditor_matches_summary_{timestamp}.csv"
        summary.to_csv(summary_file, index=False)
        logger.info("Summary saved to: %s", summary_file)
        logger.info("\n%s", summary.to_string(index=False))
    else:
        logger.warning("No matches found above the score cutoff of %s.", SCORE_CUTOFF)

    logger.info("Done.")


if __name__ == "__main__":
    main()
