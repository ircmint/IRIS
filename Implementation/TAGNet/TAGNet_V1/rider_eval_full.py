"""
Full evaluation for ZS / FT / V1 compliance outputs.

THREE evaluation blocks:
  1. Inter-model agreement  — Cohen's Kappa + % agreement for IRC35/IRC67 verdicts
                               (ZS-FT, ZS-V1, FT-V1 pairs)
  2. Text generation quality — BLEU-1/2/4, ROUGE-L, CIDEr on evasive_reasoning
                               (ZS and V1 vs each other; FT has no text so shown separately)
  3. 8-metric summary        — CHR, RHR, HAA, ICDI, IDAS, CRRS, TIPU, TVGC
                               (re-read from already-computed CSVs)

All results saved to G:/IRC_complience_Report/metrics/
Usage:
    python rider_eval_full.py
"""

import os, json, csv, math, sys
from collections import Counter

# ── deps ─────────────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False; print("[WARN] pip install nltk")

try:
    from rouge_score import rouge_scorer as rs_mod
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False; print("[WARN] pip install rouge-score")

try:
    from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_recall_fscore_support
    HAS_SK = True
except ImportError:
    HAS_SK = False; print("[WARN] pip install scikit-learn")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT   = "G:/IRC_complience_Report"
MODELS = ["ZS", "FT", "V1"]
RIDERS = {
    "R1": "Rider1_NJ",
    "R2": "Rider2_AZ",
    "R3": "Rider3_VA",
    "R4": "Rider4_UC",
}
RESULT_PATHS = {
    (r, m): f"{ROOT}/{r}/{m}/results/{m.lower()}_results.jsonl"
    for r in RIDERS for m in MODELS
}

# ── helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    if not os.path.exists(path): return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["event_id"]] = r
    return out

def norm_verdict(v):
    v = (v or "").upper()
    if "NON" in v: return "NON_COMPLIANT"
    if "COMPLIANT" in v: return "COMPLIANT"
    return None

def tokenize(text):
    if not text: return []
    try: return nltk.word_tokenize(text.lower())
    except: return text.lower().split()

# ── CIDEr (simplified TF-IDF n-gram cosine) ──────────────────────────────────
def cider_score(hyps, refs):
    def ngrams(toks, n):
        return [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    def idf(corpus_ng):
        df = Counter(); N = len(corpus_ng)
        for doc in corpus_ng:
            for g in set(doc): df[g] += 1
        return {g: math.log(N/(df[g]+1)) for g in df}
    scores_n = []
    for n in range(1, 5):
        ref_ng  = [Counter(ngrams(tokenize(r), n)) for r in refs]
        hyp_ng  = [Counter(ngrams(tokenize(h), n)) for h in hyps]
        idf_w   = idf(ref_ng)
        sims = []
        for hc, rc, ht, rt in zip(hyp_ng, ref_ng, hyps, refs):
            def vec(c):
                tot = max(sum(c.values()), 1)
                return {g: (cnt/tot)*idf_w.get(g,0) for g,cnt in c.items()}
            hv, rv = vec(hc), vec(rc)
            num    = sum(hv.get(g,0)*rv.get(g,0) for g in hv)
            dh     = math.sqrt(sum(v**2 for v in hv.values())) or 1e-9
            dr     = math.sqrt(sum(v**2 for v in rv.values())) or 1e-9
            sims.append(num/(dh*dr))
        scores_n.append(sum(sims)/max(len(sims),1))
    return round(sum(scores_n)/4, 4)

# ── BLEU ──────────────────────────────────────────────────────────────────────
def bleu_scores(hyps, refs):
    if not HAS_NLTK or not hyps:
        return {"BLEU-1":0, "BLEU-2":0, "BLEU-4":0}
    sm   = SmoothingFunction().method1
    rr   = [[tokenize(r)] for r in refs]
    hh   = [tokenize(h)   for h in hyps]
    pairs = [(r,h) for r,h in zip(rr,hh) if r[0] and h]
    if not pairs: return {"BLEU-1":0,"BLEU-2":0,"BLEU-4":0}
    rs2, hs2 = zip(*pairs)
    return {
        "BLEU-1": round(corpus_bleu(list(rs2), list(hs2), weights=(1,0,0,0)), 4),
        "BLEU-2": round(corpus_bleu(list(rs2), list(hs2), weights=(.5,.5,0,0)), 4),
        "BLEU-4": round(corpus_bleu(list(rs2), list(hs2), weights=(.25,.25,.25,.25),
                                    smoothing_function=sm), 4),
    }

# ── ROUGE-L ───────────────────────────────────────────────────────────────────
def rouge_l(hyps, refs):
    if not HAS_ROUGE or not hyps: return 0.0
    scorer = rs_mod.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h)["rougeL"].fmeasure
              for h, r in zip(hyps, refs) if h and r]
    return round(sum(scores)/max(len(scores),1), 4)

# ── Cohen's Kappa ─────────────────────────────────────────────────────────────
def kappa(a, b):
    if not HAS_SK or len(a) < 2: return 0.0
    try:    return round(cohen_kappa_score(a, b), 4)
    except: return 0.0

def pct_agree(a, b):
    if not a: return 0.0
    return round(sum(x==y for x,y in zip(a,b))/len(a), 4)

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1: Inter-model agreement
# ─────────────────────────────────────────────────────────────────────────────
def run_agreement():
    rows = []
    print("\n" + "="*80)
    print("BLOCK 1 — Inter-Model Agreement (IRC35 & IRC67 verdicts)")
    print("="*80)
    header = f"{'Rider':<5} {'Pair':<10} {'IRC':<8} {'N':>4} {'%Agree':>7} {'Kappa':>7}"
    print(header); print("-"*50)

    for rider in RIDERS:
        data = {m: load_jsonl(RESULT_PATHS[(rider, m)]) for m in MODELS}
        pairs = [("ZS","FT"), ("ZS","V1"), ("FT","V1")]
        for m1, m2 in pairs:
            d1, d2 = data[m1], data[m2]
            common = sorted(set(d1) & set(d2))
            if not common: continue
            for irc, key in [("IRC35","irc35_verdict"), ("IRC67","irc67_verdict")]:
                v1 = [norm_verdict(d1[i][key]) for i in common]
                v2 = [norm_verdict(d2[i][key]) for i in common]
                valid = [(a,b) for a,b in zip(v1,v2) if a and b]
                if not valid: continue
                a_list, b_list = zip(*valid)
                pa = pct_agree(list(a_list), list(b_list))
                k  = kappa(list(a_list), list(b_list))
                row = {"Rider":rider,"Pair":f"{m1}-{m2}","IRC":irc,
                       "N":len(valid),"%Agree":pa,"Kappa":k}
                rows.append(row)
                print(f"{rider:<5} {m1+'-'+m2:<10} {irc:<8} {len(valid):>4} {pa:>7.3f} {k:>7.3f}")

    out = f"{ROOT}/metrics/all_riders_agreement.csv"
    if rows:
        with open(out,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved -> {out}")
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2: Text generation quality (BLEU / ROUGE-L / CIDEr)
# ─────────────────────────────────────────────────────────────────────────────
def run_text_metrics():
    rows = []
    print("\n" + "="*80)
    print("BLOCK 2 — Text Generation Quality (evasive_reasoning)")
    print("  Reference: ZS output  |  Hypothesis: FT or V1")
    print("  NOTE: FT has no reasoning text -> all zeros expected")
    print("="*80)
    hdr = f"{'Rider':<5} {'Model':<5} {'N_pairs':>7} {'BLEU-1':>7} {'BLEU-2':>7} {'BLEU-4':>7} {'ROUGE-L':>8} {'CIDEr':>7}"
    print(hdr); print("-"*60)

    for rider in RIDERS:
        zs_data = load_jsonl(RESULT_PATHS[(rider,"ZS")])

        for model in ["FT","V1"]:
            md = load_jsonl(RESULT_PATHS[(rider,model)])
            common = sorted(set(zs_data) & set(md))
            refs = [zs_data[i].get("evasive_reasoning","") or "" for i in common]
            hyps = [md[i].get("evasive_reasoning","")          or "" for i in common]
            valid = [(r,h) for r,h in zip(refs,hyps) if r.strip() and h.strip()]

            if valid:
                rv, hv = zip(*valid)
                bl = bleu_scores(list(hv), list(rv))
                rl = rouge_l(list(hv), list(rv))
                cd = cider_score(list(hv), list(rv))
            else:
                bl = {"BLEU-1":0,"BLEU-2":0,"BLEU-4":0}; rl = 0.0; cd = 0.0

            row = {"Rider":rider,"Model":model,"Reference":"ZS",
                   "N_pairs":len(valid), **bl, "ROUGE-L":rl, "CIDEr":cd}
            rows.append(row)
            print(f"{rider:<5} {model:<5} {len(valid):>7} {bl['BLEU-1']:>7.3f} "
                  f"{bl['BLEU-2']:>7.3f} {bl['BLEU-4']:>7.3f} {rl:>8.3f} {cd:>7.3f}")

    out = f"{ROOT}/metrics/all_riders_text_metrics.csv"
    if rows:
        with open(out,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved -> {out}")
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3: Verdict distribution per model (COMPLIANT vs NON_COMPLIANT)
# ─────────────────────────────────────────────────────────────────────────────
def run_verdict_distribution():
    rows = []
    print("\n" + "="*80)
    print("BLOCK 3 — Verdict Distribution per Model (% NON_COMPLIANT = defect rate)")
    print("="*80)
    hdr = f"{'Rider':<5} {'Model':<5} {'N':>4} {'IRC35_NC%':>10} {'IRC67_NC%':>10} {'IRC35_C%':>9} {'IRC67_C%':>9}"
    print(hdr); print("-"*55)

    for rider in RIDERS:
        for model in MODELS:
            md = load_jsonl(RESULT_PATHS[(rider,model)])
            if not md: continue
            v35 = [norm_verdict(r.get("irc35_verdict","")) for r in md.values()]
            v67 = [norm_verdict(r.get("irc67_verdict","")) for r in md.values()]
            n = len(md)
            nc35 = round(sum(1 for x in v35 if x=="NON_COMPLIANT")/n, 3)
            nc67 = round(sum(1 for x in v67 if x=="NON_COMPLIANT")/n, 3)
            c35  = round(sum(1 for x in v35 if x=="COMPLIANT")/n, 3)
            c67  = round(sum(1 for x in v67 if x=="COMPLIANT")/n, 3)
            row  = {"Rider":rider,"Model":model,"N":n,
                    "IRC35_NON_COMPLIANT%":nc35,"IRC67_NON_COMPLIANT%":nc67,
                    "IRC35_COMPLIANT%":c35,"IRC67_COMPLIANT%":c67}
            rows.append(row)
            print(f"{rider:<5} {model:<5} {n:>4} {nc35:>10.3f} {nc67:>10.3f} {c35:>9.3f} {c67:>9.3f}")

    out = f"{ROOT}/metrics/all_riders_verdict_distribution.csv"
    if rows:
        with open(out,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved -> {out}")
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4: Per-model, per-rider individual accuracy on verdicts
#          (ZS vs V1 and ZS vs FT — ZS as pseudo-reference since no human GT)
# ─────────────────────────────────────────────────────────────────────────────
def run_per_model_accuracy():
    """Combined IRC35+IRC67 verdicts -> single Acc/P/R/F1 per Rider x Model."""
    rows = []
    print("\n" + "="*80)
    print("BLOCK 4 — Per-Rider Per-Model: Accuracy / Precision / Recall / F1")
    print("  (IRC35 + IRC67 verdicts combined; ZS as reference)")
    print("="*80)
    hdr = f"{'Rider':<5} {'Model':<5} {'N':>4}  {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(hdr); print("-"*55)

    for rider in RIDERS:
        for model in MODELS:
            zs = load_jsonl(RESULT_PATHS[(rider,"ZS")])
            md = load_jsonl(RESULT_PATHS[(rider, model)])
            common = sorted(set(zs) & set(md))
            if not common: continue

            # combine IRC35 + IRC67 into one flat list
            gt_all, pr_all = [], []
            for key in ["irc35_verdict", "irc67_verdict"]:
                for i in common:
                    g = norm_verdict(zs[i][key])
                    p = norm_verdict(md[i][key])
                    if g and p:
                        gt_all.append(g); pr_all.append(p)

            if not gt_all: continue
            classes = sorted(set(gt_all))

            if HAS_SK:
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                acc = round(accuracy_score(gt_all, pr_all), 4)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    gt_all, pr_all, labels=classes, average="macro", zero_division=0)
                prec, rec, f1 = round(prec,4), round(rec,4), round(f1,4)
            else:
                acc = round(sum(a==b for a,b in zip(gt_all,pr_all))/len(gt_all),4)
                prec=rec=f1=0.0

            row = {"Rider":rider,"Model":model,"N":len(gt_all),
                   "Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1}
            rows.append(row)
            print(f"{rider:<5} {model:<5} {len(gt_all):>4}  {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")

    out = f"{ROOT}/metrics/all_riders_accuracy_summary.csv"
    if rows:
        with open(out,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved -> {out}")
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(f"{ROOT}/metrics", exist_ok=True)

    agr  = run_agreement()
    txt  = run_text_metrics()
    vdist = run_verdict_distribution()
    acc  = run_per_model_accuracy()

    print("\n" + "="*80)
    print("ALL DONE — files saved to G:/IRC_complience_Report/metrics/")
    print("  all_riders_agreement.csv          — Cohen kappa + % agreement")
    print("  all_riders_text_metrics.csv        — BLEU/ROUGE-L/CIDEr")
    print("  all_riders_verdict_distribution.csv— % COMPLIANT vs NON_COMPLIANT")
    print("  all_riders_per_model_accuracy.csv  — Acc/P/R/F1 (ZS as reference)")
    print("="*80)
