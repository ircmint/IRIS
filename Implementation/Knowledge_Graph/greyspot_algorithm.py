"""
Greyspot Algorithm — Risk Analytics with Spatial Clustering
=============================================================
GPS + Evasive Behavior + DBSCAN

Implements STEP 1-5 exactly as specified:
  1. Project lat/lon -> UTM (metric coordinates) so eps is a true Euclidean distance
  2. DBSCAN clustering on projected (x, y)
  3. Per-cluster metrics: n_events, observed_risk_rate, BCS, same_type_share, infrastructure_share
  4. Composite GreyspotScore
  5. Rank + flag low-confidence clusters

Expected input columns (rename via COLUMN_MAP below if your CSVs differ):
    lat, lon, ICDI_final, severity, violation_type, presence
    ('presence' = "Yes"/"No" -> whether road infrastructure was present at the event)

Usage:
    python greyspot_algorithm.py --input gold_candidates.csv --eps 75 --min_samples 4
"""

import argparse
import sys
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    sys.exit("Missing dependency: pip install scikit-learn --break-system-packages")

try:
    import pyproj
except ImportError:
    sys.exit("Missing dependency: pip install pyproj --break-system-packages")


# --------------------------------------------------------------------------
# CONFIG — adjust to match your actual CSV headers
# --------------------------------------------------------------------------
COLUMN_MAP = {
    "lat": "lat",
    "lon": "lon",
    "icdi": "ICDI_final",
    "severity": "severity",
    "violation_type": "violation_type",
    "presence": "presence",          # "Yes" = infrastructure present, "No" = absent
}

DEFAULT_PARAMS = dict(
    eps=75.0,              # meters
    min_samples=4,
    k=10,                  # BCS shrinkage strength (pseudo-count toward global prior)
    beta=0.5,              # repeatability weight in GreyspotScore
    gamma=0.5,             # infrastructure-vs-behavior weighting in GreyspotScore
    min_reportable_n=5,    # clusters smaller than this are flagged low-confidence
)


# --------------------------------------------------------------------------
# STEP 1 — Project lat/lon to UTM metric coordinates
# --------------------------------------------------------------------------
def project_to_utm(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Convert WGS84 lat/lon to UTM (x, y) in meters.
    UTM zone is auto-detected from the mean longitude of the dataset so that
    all events fall in a single locally-accurate projected CRS.
    """
    mean_lon = df[lon_col].mean()
    mean_lat = df[lat_col].mean()

    utm_zone = int((mean_lon + 180) / 6) + 1
    hemisphere = "north" if mean_lat >= 0 else "south"
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x, y = transformer.transform(df[lon_col].values, df[lat_col].values)

    out = df.copy()
    out["x"] = x
    out["y"] = y
    return out


# --------------------------------------------------------------------------
# STEP 2 — DBSCAN on projected coordinates
# --------------------------------------------------------------------------
def run_dbscan(df: pd.DataFrame, eps: float, min_samples: int) -> pd.DataFrame:
    """
    eps is in meters because x, y are already UTM meters — no unit conversion needed.
    Returns df with an added 'cluster' column (-1 = noise).
    """
    coords = df[["x", "y"]].to_numpy()
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(coords)
    out = df.copy()
    out["cluster"] = labels
    return out


# --------------------------------------------------------------------------
# STEP 3 — Per-cluster metrics
# --------------------------------------------------------------------------
def n_events(cluster_df: pd.DataFrame) -> int:
    """3a. Count of events in the cluster."""
    return len(cluster_df)


def observed_risk_rate(cluster_df: pd.DataFrame, icdi_col: str) -> float:
    """3b. Mean ICDI_final across events in the cluster."""
    return cluster_df[icdi_col].mean()


def bayesian_cluster_score(n: int, risk_rate: float, k: float, global_prior_icdi: float) -> float:
    """
    3c. BCS(c) — shrinkage estimator that pulls small/noisy clusters toward
    the global prior risk rate. Larger clusters trust their own observed
    risk_rate more; smaller clusters get pulled toward the prior.
    """
    return (n * risk_rate + k * global_prior_icdi) / (n + k)


def same_type_share(cluster_df: pd.DataFrame, type_col: str):
    """
    3d. Repeatability signal — fraction of the cluster's events that share
    the single most common (dominant) violation_type in that cluster.
    Returns (dominant_type, share).
    """
    counts = cluster_df[type_col].value_counts()
    dominant_type = counts.idxmax()
    share = counts.max() / len(cluster_df)
    return dominant_type, share


def infrastructure_share(cluster_df: pd.DataFrame, icdi_col: str, presence_col: str):
    """
    3e. Fraction of the cluster's ICDI-weighted risk attributable to events
    where presence == "No" (i.e. no infrastructure was present -> the road/
    environment itself is implicated, not just rider behavior).

    Returns np.nan if the cluster has no usable Yes/No presence data (e.g.
    presence == "Unknown" for every event) rather than silently defaulting
    to 0 or 1, since that would misrepresent an infra-vs-behavior split
    we don't actually have evidence for.
    """
    known = cluster_df[cluster_df[presence_col].isin(["Yes", "No"])]
    if known.empty:
        return np.nan

    total_risk = known[icdi_col].sum()
    if total_risk == 0:
        # Fall back to a simple event-count share if there's no risk mass to weight by
        return (known[presence_col] == "No").mean()
    infra_absent_risk = known.loc[known[presence_col] == "No", icdi_col].sum()
    return infra_absent_risk / total_risk


# --------------------------------------------------------------------------
# STEP 4 — Composite GreyspotScore
# --------------------------------------------------------------------------
def greyspot_score(bcs: float, same_type_share_val: float, infra_share: float,
                    beta: float, gamma: float) -> float:
    """
    GreyspotScore(c) = BCS(c)
                        * (1 + beta * same_type_share(c))
                        * [gamma * infra_share(c) + (1-gamma) * (1 - infra_share(c))]

    - The repeatability term boosts clusters where the same violation type
      keeps recurring (a real behavioral/infra pattern, not noise).
    - The bracket term is a tunable blend: gamma=1 favors pure infrastructure
      spots, gamma=0 favors pure behavior spots, gamma=0.5 is neutral.

    If infra_share is NaN (no infrastructure-presence data available for
    this cluster), the blend term falls back to a neutral 0.5 so the score
    is still computable — but this means the score is NOT actually infra-
    aware for that cluster. Check the infrastructure_share column before
    trusting the infra/behavior split.
    """
    repeatability_term = 1 + beta * same_type_share_val
    if pd.isna(infra_share):
        blend_term = 0.5
    else:
        blend_term = gamma * infra_share + (1 - gamma) * (1 - infra_share)
    return bcs * repeatability_term * blend_term


# --------------------------------------------------------------------------
# STEP 3+4 driver — compute all per-cluster metrics and composite score
# --------------------------------------------------------------------------
def compute_cluster_metrics(df: pd.DataFrame, cols: dict, k: float, beta: float, gamma: float) -> pd.DataFrame:
    icdi_col = cols["icdi"]
    type_col = cols["violation_type"]
    presence_col = cols["presence"]

    global_prior_icdi = df[icdi_col].mean()

    rows = []
    for cluster_id, cluster_df in df[df["cluster"] != -1].groupby("cluster"):
        n = n_events(cluster_df)
        risk_rate = observed_risk_rate(cluster_df, icdi_col)
        bcs = bayesian_cluster_score(n, risk_rate, k, global_prior_icdi)
        dom_type, sts = same_type_share(cluster_df, type_col)
        infra_share = infrastructure_share(cluster_df, icdi_col, presence_col)
        score = greyspot_score(bcs, sts, infra_share, beta, gamma)

        rows.append({
            "cluster": cluster_id,
            "cluster_centroid_lat": cluster_df["lat_orig"].mean(),
            "cluster_centroid_lon": cluster_df["lon_orig"].mean(),
            "n_events": n,
            "observed_risk_rate": risk_rate,
            "BCS": bcs,
            "dominant_violation_type": dom_type,
            "same_type_share": sts,
            "infrastructure_share": infra_share,
            "behavior_share": 1 - infra_share,
            "GreyspotScore": score,
        })

    result = pd.DataFrame(rows)
    return result, global_prior_icdi


# --------------------------------------------------------------------------
# STEP 5 — Rank + flag low-confidence clusters
# --------------------------------------------------------------------------
def rank_and_flag(cluster_metrics: pd.DataFrame, min_reportable_n: int) -> pd.DataFrame:
    out = cluster_metrics.sort_values("GreyspotScore", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out["confidence_flag"] = np.where(
        out["n_events"] < min_reportable_n,
        "LOW-CONFIDENCE (needs more data)",
        "OK"
    )
    return out


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------
def run_greyspot_pipeline(df: pd.DataFrame, cols: dict = COLUMN_MAP, params: dict = None) -> pd.DataFrame:
    p = {**DEFAULT_PARAMS, **(params or {})}

    # keep originals for centroid reporting in lat/lon (not UTM)
    df = df.rename(columns={cols["lat"]: "lat_orig", cols["lon"]: "lon_orig"})

    # STEP 1
    df = project_to_utm(df, "lat_orig", "lon_orig")

    # STEP 2
    df = run_dbscan(df, eps=p["eps"], min_samples=p["min_samples"])

    n_noise = (df["cluster"] == -1).sum()
    n_clusters = df.loc[df["cluster"] != -1, "cluster"].nunique()
    print(f"DBSCAN: {n_clusters} clusters found, {n_noise} events labeled noise "
          f"(eps={p['eps']}m, min_samples={p['min_samples']})")

    # STEP 3 + 4
    cluster_metrics, global_prior = compute_cluster_metrics(
        df, cols, k=p["k"], beta=p["beta"], gamma=p["gamma"]
    )
    print(f"Global prior ICDI (dataset-wide mean risk): {global_prior:.4f}")

    # STEP 5
    ranked = rank_and_flag(cluster_metrics, min_reportable_n=p["min_reportable_n"])
    return ranked


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Greyspot risk-analytics clustering pipeline")
    parser.add_argument("--input", required=True, help="Path to event-level CSV "
                         "(needs lat/lon/ICDI_final/severity/violation_type/presence columns)")
    parser.add_argument("--output", default="greyspot_ranked.csv")
    parser.add_argument("--eps", type=float, default=DEFAULT_PARAMS["eps"], help="DBSCAN radius, meters")
    parser.add_argument("--min_samples", type=int, default=DEFAULT_PARAMS["min_samples"])
    parser.add_argument("--k", type=float, default=DEFAULT_PARAMS["k"], help="BCS shrinkage strength")
    parser.add_argument("--beta", type=float, default=DEFAULT_PARAMS["beta"])
    parser.add_argument("--gamma", type=float, default=DEFAULT_PARAMS["gamma"])
    parser.add_argument("--min_reportable_n", type=int, default=DEFAULT_PARAMS["min_reportable_n"])
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    params = dict(
        eps=args.eps, min_samples=args.min_samples, k=args.k,
        beta=args.beta, gamma=args.gamma, min_reportable_n=args.min_reportable_n,
    )

    ranked = run_greyspot_pipeline(df, COLUMN_MAP, params)
    ranked.to_csv(args.output, index=False)
    print(f"\nSaved {len(ranked)} ranked clusters -> {args.output}")
    print(ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()