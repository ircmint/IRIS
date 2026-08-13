# step2_build_gold.py
# ---------------------------------------------------------------------------
# Reads your reviewed candidate_events.csv (decision filled in), keeps only
# 'confirm' rows, uses adjusted_start/adjusted_end where you provided them,
# and writes gold_labels_v1_FROZEN.csv in clip_id,start_time,end_time format.
#
# Run:  python step2_build_gold.py
# ---------------------------------------------------------------------------
import os
import pandas as pd
import config


def build_gold():
    if not os.path.exists(config.CANDIDATE_EVENTS_CSV):
        raise FileNotFoundError(
            f"{config.CANDIDATE_EVENTS_CSV} not found. Run step1_build_candidates.py "
            f"and review it first."
        )
    df = pd.read_csv(config.CANDIDATE_EVENTS_CSV)
    df["decision"] = df["decision"].astype(str).str.strip().str.lower()
    confirmed = df[df["decision"] == "confirm"].copy()

    if confirmed.empty:
        raise ValueError(
            "No rows with decision == 'confirm' found. "
            "Did you finish the review pass in candidate_events.csv?"
        )

    def pick(row, base, adj):
        val = row.get(adj, "")
        if pd.notna(val) and str(val).strip() != "":
            return float(val)
        return float(row[base])

    confirmed["final_start"] = confirmed.apply(lambda r: pick(r, "start_time", "adjusted_start"), axis=1)
    confirmed["final_end"] = confirmed.apply(lambda r: pick(r, "end_time", "adjusted_end"), axis=1)

    out = confirmed[["clip_id", "final_start", "final_end"]].rename(
        columns={"final_start": "start_time", "final_end": "end_time"}
    )
    os.makedirs(config.DATA_DIR, exist_ok=True)
    out.to_csv(config.GOLD_LABELS_CSV, index=False)
    print(f"Wrote {len(out)} confirmed gold events to:")
    print(f"  {config.GOLD_LABELS_CSV}")


if __name__ == "__main__":
    build_gold()
