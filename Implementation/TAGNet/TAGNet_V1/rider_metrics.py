"""
Compute 8 metrics (CHR, RHR, HAA, ICDI, IDAS, CRRS, TIPU, TVGC) for all 4 riders.
Reads ZS/FT/V1 result JSONLs produced by rider_pipeline.py.

Usage:
    python rider_metrics.py --rider R1|R2|R3|R4|ALL

Outputs:
    G:/Metrics/R{n}_metrics_summary.csv  — 3-row summary (ZS/FT/V1)
    G:/Metrics/R{n}_metrics_per_event.csv
    G:/Metrics/all_riders_combined.csv   — 12-row combined table for paper
"""

import os, json, re, csv, argparse
import numpy as np
from sentence_transformers import SentenceTransformer, util as st_util

EMBED = SentenceTransformer("all-MiniLM-L6-v2")  # local install, runs fine on Windows

ACTION_EMBED = {
    "Acceleration":  EMBED.encode("sudden acceleration gap in traffic throttle speed up"),
    "Deceleration":  EMBED.encode("deceleration slowing braking vehicle ahead following"),
    "Hard_Braking":  EMBED.encode("emergency hard braking sudden obstacle collision avoidance cut off"),
    "Lane_Change":   EMBED.encode("lane change swerve avoid obstacle lateral overtake merge"),
}
HAZARD_KW  = r"(?i)(vehicle|motorcy|bike|auto|truck|bus|swerv|brake|obstacle|hazard|pothole|pedestrian|sudden|cut.?off|car|lorry|oncoming)"
ROAD_BLEED = r"(?i)(road surface|IRC[:-]?3[57]|IRC[:-]?6[67]|infrastructure|pavement)"
TEMPLATE   = [
    "The rider accelerated rapidly due to a gap opening in traffic ahead",
    "A slowing vehicle ahead caused the rider to decelerate sharply",
    "An abrupt obstacle or vehicle cutting into the lane required emergency hard braking",
    "A slow or stationary vehicle in the current lane required a lateral lane change",
]
ACTION_KW = {
    "Acceleration": r"(?i)(accelerat|speed up|throttle|gap|pull away)",
    "Deceleration": r"(?i)(decelerat|slow|brake|following distance|slow.?down)",
    "Hard_Braking": r"(?i)(hard brak|emergency|sudden stop|collision|obstacle|cut.?off|brake)",
    "Lane_Change":  r"(?i)(lane change|swerv|lateral|overtake|merge)",
}
ENV_ACTION = {
    "Acceleration": r"(?i)(vehicle|traffic|car|bike|auto|moving|ahead)",
    "Deceleration": r"(?i)(vehicle|traffic|car|ahead|slow|queue|stop)",
    "Hard_Braking": r"(?i)(vehicle|traffic|car|bike|ahead|cut|obstacle|sudden)",
    "Lane_Change":  r"(?i)(vehicle|traffic|car|ahead|lane|block|slow)",
}
TEL_KW = r"(?i)(speed|kmh|km/h|accelerat|decelerat|lateral|brake|km|velocity)"

ROOT = "G:/IRC_complience_Report"
RIDER_PATHS = {
    "R1": {
        "name": "Rider1_NJ",
        "ZS": f"{ROOT}/R1/ZS/results/zs_results.jsonl",
        "FT": f"{ROOT}/R1/FT/results/ft_results.jsonl",
        "V1": f"{ROOT}/R1/V1/results/v1_results.jsonl",
        "out": f"{ROOT}/metrics/R1",
    },
    "R2": {
        "name": "Rider2_AZ",
        "ZS": f"{ROOT}/R2/ZS/results/zs_results.jsonl",
        "FT": f"{ROOT}/R2/FT/results/ft_results.jsonl",
        "V1": f"{ROOT}/R2/V1/results/v1_results.jsonl",
        "out": f"{ROOT}/metrics/R2",
    },
    "R3": {
        "name": "Rider3_VA",
        "ZS": f"{ROOT}/R3/ZS/results/zs_results.jsonl",
        "FT": f"{ROOT}/R3/FT/results/ft_results.jsonl",
        "V1": f"{ROOT}/R3/V1/results/v1_results.jsonl",
        "out": f"{ROOT}/metrics/R3",
    },
    "R4": {
        "name": "Rider4_UC",
        "ZS": f"{ROOT}/R4/ZS/results/zs_results.jsonl",
        "FT": f"{ROOT}/R4/FT/results/ft_results.jsonl",
        "V1": f"{ROOT}/R4/V1/results/v1_results.jsonl",
        "out": f"{ROOT}/metrics/R4",
    },
    "Pilot": {
        "name": "Pilot_GX019940",
        "ZS": f"{ROOT}/Pilot/ZS/results/zs_results.jsonl",
        "FT": f"{ROOT}/Pilot/FT/results/ft_results.jsonl",
        "V1": f"{ROOT}/Pilot/V1/results/v1_results.jsonl",
        "out": f"{ROOT}/metrics/Pilot",
    },
    "Motor": {
        "name": "Motor_GX0422",
        "ZS": f"{ROOT}/Motor/ZS/results/zs_results.jsonl",
        "FT": f"{ROOT}/Motor/FT/results/ft_results.jsonl",
        "V1": f"{ROOT}/Motor/V1/results/v1_results.jsonl",
        "out": f"{ROOT}/metrics/Motor",
    },
}

def load_jsonl(path):
    if not os.path.exists(path): return []
    return [json.loads(l) for l in open(path) if l.strip()]

def compute(rows, model):
    n = len(rows)
    if n == 0: return None, []
    CHR=RHR=HAA_s=ICDI_s=IDAS=CRRS_s=TIPU=TVGC=0
    per=[]
    for r in rows:
        action=r.get("evasive_action",""); evas=r.get("evasive_reasoning","")
        env=r.get("env",""); road=r.get("road_surface",""); infra=r.get("infrastructure","")
        ret35=set(x.strip() for x in r.get("retrieved_irc35","").split(";") if x.strip())
        ret67=set(x.strip() for x in r.get("retrieved_irc67","").split(";") if x.strip())
        c35=r.get("irc35_clause","").strip(); c67=r.get("irc67_clause","").strip()
        is_tmpl=any(t[:35] in evas for t in TEMPLATE)
        has_haz=bool(re.search(HAZARD_KW,evas))
        has_bld=bool(re.search(ROAD_BLEED,evas))

        chr_v  = 1 if (bool(c35) and c35 in ret35) or (bool(c67) and c67 in ret67) else 0
        rhr_v  = 0 if is_tmpl or not has_haz or has_bld or len(evas.split())<8 else 1
        haa_v  = 0.0
        if evas and action in ACTION_EMBED:
            haa_v=round(float(st_util.cos_sim(EMBED.encode(evas), ACTION_EMBED[action])),4)
        icdi_v = 1 if "NON" in r.get("irc35_verdict","") or "NON" in r.get("irc67_verdict","") else 0
        idas_v = 1 if re.search(ACTION_KW.get(action,"x^"), evas) else 0
        crrs_v = round((r.get("crrs35",0)+r.get("crrs67",0))/2, 4)
        tipu_v = 1 if model=="V1" and re.search(TEL_KW," ".join([env,evas,road,infra])) else 0
        tvgc_v = 1 if re.search(ENV_ACTION.get(action,"x^"), env) else 0

        CHR+=chr_v; RHR+=rhr_v; HAA_s+=haa_v; ICDI_s+=icdi_v
        IDAS+=idas_v; CRRS_s+=crrs_v; TIPU+=tipu_v; TVGC+=tvgc_v
        per.append({
            "model":model,"rider":r.get("rider",""),"location":r.get("location",""),
            "event_id":r.get("event_id",""),"evasive_action":action,
            "irc35_verdict":r.get("irc35_verdict",""),"irc67_verdict":r.get("irc67_verdict",""),
            "irc35_clause":c35,"irc67_clause":c67,
            "retrieved_irc35":r.get("retrieved_irc35",""),"retrieved_irc67":r.get("retrieved_irc67",""),
            "crrs35":r.get("crrs35",0),"crrs67":r.get("crrs67",0),
            "CHR":chr_v,"RHR":rhr_v,"HAA":haa_v,"ICDI":icdi_v,
            "IDAS":idas_v,"CRRS":crrs_v,"TIPU":tipu_v,"TVGC":tvgc_v,
            "env":env,"evasive_reasoning":evas,"road_surface":road,"infrastructure":infra,
        })
    summary={"Rider":r.get("rider",""),"Location":r.get("location",""),"Model":model,"N":n,
              "CHR":round(CHR/n,4),"RHR":round(RHR/n,4),"HAA":round(HAA_s/n,4),
              "ICDI":round(ICDI_s/n,4),"IDAS":round(IDAS/n,4),"CRRS":round(CRRS_s/n,4),
              "TIPU":round(TIPU/n,4),"TVGC":round(TVGC/n,4)}
    return summary, per

def run_rider(rider_key):
    cfg = RIDER_PATHS[rider_key]
    os.makedirs(cfg["out"], exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Metrics for {cfg['name']} ({rider_key})")
    print(f"{'='*70}")

    summaries=[]; per_all=[]
    for model in ["ZS","FT","V1"]:
        rows = load_jsonl(cfg[model])
        print(f"  [{model}] {len(rows)} events")
        if not rows: continue
        s, p = compute(rows, model)
        if s: summaries.append(s); per_all.extend(p)

    if not summaries:
        print(f"  [SKIP] No results found yet for {rider_key}")
        return summaries

    # Print summary table
    print(f"\n{'Model':<6} {'N':>4} {'CHR':>6} {'RHR':>6} {'HAA':>6} {'ICDI':>6} {'IDAS':>6} {'CRRS':>6} {'TIPU':>6} {'TVGC':>6}")
    print("-"*70)
    for m in summaries:
        print(f"{m['Model']:<6} {m['N']:>4} {m['CHR']:>6.3f} {m['RHR']:>6.3f} {m['HAA']:>6.3f} "
              f"{m['ICDI']:>6.3f} {m['IDAS']:>6.3f} {m['CRRS']:>6.3f} {m['TIPU']:>6.3f} {m['TVGC']:>6.3f}")

    # Write CSVs (with fallback suffix if file is locked)
    def safe_open(path, *a, **kw):
        try:
            return open(path, *a, **kw)
        except PermissionError:
            alt = path.replace(".csv", "_new.csv")
            print(f"  [WARN] {path} locked, writing to {alt}")
            return open(alt, *a, **kw)

    sum_csv = f"{cfg['out']}/{rider_key}_metrics_summary.csv"
    per_csv = f"{cfg['out']}/{rider_key}_metrics_per_event.csv"
    fields_s = ["Rider","Location","Model","N","CHR","RHR","HAA","ICDI","IDAS","CRRS","TIPU","TVGC"]
    with safe_open(sum_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields_s); w.writeheader(); w.writerows(summaries)
    if per_all:
        with safe_open(per_csv,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(per_all[0].keys())); w.writeheader(); w.writerows(per_all)
    print(f"  Summary -> {sum_csv}")
    print(f"  Per-event -> {per_csv}")
    return summaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rider", default="ALL")
    args = parser.parse_args()

    keys = list(RIDER_PATHS.keys()) if args.rider.upper()=="ALL" else [
        next((k for k in RIDER_PATHS if k.upper()==args.rider.upper()), args.rider)]
    all_summaries = []
    for k in keys:
        all_summaries.extend(run_rider(k))

    # Write combined table
    if all_summaries:
        combined = f"{ROOT}/metrics/all_riders_combined.csv"
        os.makedirs(f"{ROOT}/metrics", exist_ok=True)
        fields = ["Rider","Location","Model","N","CHR","RHR","HAA","ICDI","IDAS","CRRS","TIPU","TVGC"]
        with open(combined,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(all_summaries)
        print(f"\nCombined -> {combined}")

        print(f"\n{'Rider':<12} {'Location':<25} {'Model':<6} {'N':>4} {'CHR':>6} {'RHR':>6} {'HAA':>6} {'ICDI':>6} {'IDAS':>6} {'CRRS':>6} {'TIPU':>6} {'TVGC':>6}")
        print("="*110)
        for m in all_summaries:
            print(f"{m['Rider']:<12} {m['Location']:<25} {m['Model']:<6} {m['N']:>4} "
                  f"{m['CHR']:>6.3f} {m['RHR']:>6.3f} {m['HAA']:>6.3f} "
                  f"{m['ICDI']:>6.3f} {m['IDAS']:>6.3f} {m['CRRS']:>6.3f} "
                  f"{m['TIPU']:>6.3f} {m['TVGC']:>6.3f}")
    print("\nDone.")
