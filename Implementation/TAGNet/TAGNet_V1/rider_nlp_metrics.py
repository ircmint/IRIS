"""
NLP + Classification metrics for ZS/FT/V1 compliance outputs.

Ground truth: per-rider dataset JSONs from gold_candidates (human-reviewed).
  - evasive_action GT: from dataset JSON (evasive_action field per event_id)
  - reasoning GT: from gold_candidates notes field (human annotation)

Classification (evasive_action):
  - GT = gold_candidates confirmed evasive_action labels
  - Evaluates ALL 3 models: ZS, FT, V1
  - Metrics: Accuracy, Precision, Recall, F1 (macro)

Text generation (evasive_reasoning):
  - Reference = gold_candidates notes (human description of event)
  - Metrics: BLEU-1/2/4, ROUGE-L, CIDEr for ZS, FT, V1

Usage:
    python rider_nlp_metrics.py --rider ALL
"""

import os, json, re, csv, argparse, math
from collections import Counter, defaultdict

# ── libraries ────────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("[WARN] nltk not found: pip install nltk")

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[WARN] rouge_score not found: pip install rouge-score")

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    HAS_SK = True
except ImportError:
    HAS_SK = False
    print("[WARN] sklearn not found: pip install scikit-learn")

# ── config ───────────────────────────────────────────────────────────────────
ROOT    = "G:/IRC_complience_Report"
MODELS  = ["ZS", "FT", "V1"]

RIDER_PATHS = {
    "R1": {
        "name": "Rider1_NJ",
        "dataset": "G:/ZS-Compliance-Pipeline/R1/Rider1_NJ_dataset.json",
        "gold_csv": "G:/Driver_Behaviour/pipeline/Custom_Data/Rider1_NJ/gold_candidates.csv",
        "ZS": f"{ROOT}/R1/ZS/results/zs_results.jsonl",
        "FT": f"{ROOT}/R1/FT/results/ft_results.jsonl",
        "V1": f"{ROOT}/R1/V1/results/v1_results.jsonl",
        "out": f"{ROOT}/metrics/R1",
    },
    "R2": {
        "name": "Rider2_AZ",
        "dataset": "G:/ZS-Compliance-Pipeline/R2/Rider2_AZ_dataset.json",
        "gold_csv": "G:/Driver_Behaviour/pipeline/Custom_Data/Rider2_AZ/gold_candidates.csv",
        "ZS": f"{ROOT}/R2/ZS/results/zs_results.jsonl",
        "FT": f"{ROOT}/R2/FT/results/ft_results.jsonl",
        "V1": f"{ROOT}/R2/V1/results/v1_results.jsonl",
        "out": f"{ROOT}/metrics/R2",
    },
    "R3": {
        "name": "Rider3_VA",
        "dataset": "G:/ZS-Compliance-Pipeline/R3/Rider3_VA_dataset.json",
        "gold_csv": "G:/Driver_Behaviour/pipeline/Custom_Data/Rider3_VA/gold_candidates.csv",
        "ZS": f"{ROOT}/R3/ZS/results/zs_results.jsonl",
        "FT": f"{ROOT}/R3/FT/results/ft_results.jsonl",
        "V1": f"{ROOT}/R3/V1/results/v1_results.jsonl",
        "out": f"{ROOT}/metrics/R3",
    },
    "R4": {
        "name": "Rider4_UC",
        "dataset": "G:/ZS-Compliance-Pipeline/R4/Rider4_UC_dataset.json",
        "gold_csv": "G:/Driver_Behaviour/pipeline/Custom_Data/Rider4_UC/gold_candidates.csv",
        "ZS": f"{ROOT}/R4/ZS/results/zs_results.jsonl",
        "FT": f"{ROOT}/R4/FT/results/ft_results.jsonl",
        "V1": f"{ROOT}/R4/V1/results/v1_results.jsonl",
        "out": f"{ROOT}/metrics/R4",
    },
}

# Map gold_candidates category -> model output evasive_action
ACTION_MAP = {
    "acceleration":     "Acceleration",
    "deceleration":     "Deceleration",
    "lane_change":      "Lane_Change",
    "braking":          "Hard_Braking",
    "hard_braking":     "Hard_Braking",
    "emergency_swerve": "Hard_Braking",
    "zigzag":           "Lane_Change",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    if not os.path.exists(path): return {}
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["event_id"]] = r
    return rows

def load_dataset_gt(path):
    """Load per-event GT evasive_action and notes from dataset JSON."""
    if not os.path.exists(path): return {}
    data = json.load(open(path, encoding="utf-8"))
    gt = {}
    for item in data:
        eid = item["event_id"]
        gt[eid] = {
            "action": item.get("evasive_action", ""),
            "notes":  item.get("notes", ""),
        }
    return gt

def normalize_action(a):
    """Normalize action label for comparison."""
    a = (a or "").strip()
    mapping = {
        "Acceleration": "Acceleration",
        "Deceleration": "Deceleration",
        "Hard_Braking": "Hard_Braking",
        "Lane_Change": "Lane_Change",
        "acceleration": "Acceleration",
        "deceleration": "Deceleration",
        "hard_braking": "Hard_Braking",
        "braking": "Hard_Braking",
        "lane_change": "Lane_Change",
        "emergency_swerve": "Hard_Braking",
        "zigzag": "Lane_Change",
    }
    return mapping.get(a, a)

def tokenize(text):
    if not text: return []
    try:
        return nltk.word_tokenize(text.lower())
    except:
        return text.lower().split()

# ── CIDEr (simplified, document-level TF-IDF) ────────────────────────────────
def compute_cider(hyp_list, ref_list):
    """Simplified CIDEr-D (n=1..4 ngrams, TF-IDF weighted)."""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def tfidf_weights(corpus_ngrams):
        df = Counter()
        for doc in corpus_ngrams:
            for ng in set(doc.keys()):
                df[ng] += 1
        N = len(corpus_ngrams)
        return {ng: math.log(N / (df[ng]+1)) for ng in df}

    def vec(tokens, idf, n):
        ngrams = get_ngrams(tokens, n)
        tf = Counter(ngrams)
        total = max(len(ngrams), 1)
        return {ng: (cnt/total) * idf.get(ng, 0) for ng, cnt in tf.items()}

    scores = []
    for n in range(1, 5):
        ref_ngrams = [Counter(get_ngrams(tokenize(r), n)) for r in ref_list]
        hyp_ngrams = [Counter(get_ngrams(tokenize(h), n)) for h in hyp_list]
        idf = tfidf_weights(ref_ngrams)
        cider_n = []
        for h_cnt, r_cnt, h_tok, r_tok in zip(hyp_ngrams, ref_ngrams, hyp_list, ref_list):
            h_vec = vec(tokenize(h_tok), idf, n)
            r_vec = vec(tokenize(r_tok), idf, n)
            num = sum(h_vec.get(k,0)*r_vec.get(k,0) for k in h_vec)
            denom_h = math.sqrt(sum(v**2 for v in h_vec.values())) or 1e-9
            denom_r = math.sqrt(sum(v**2 for v in r_vec.values())) or 1e-9
            cider_n.append(num / (denom_h * denom_r))
        scores.append(sum(cider_n)/max(len(cider_n),1))
    return round(sum(scores)/4, 4)

# ── BLEU ──────────────────────────────────────────────────────────────────────
def compute_bleu(hyp_list, ref_list):
    if not HAS_NLTK:
        return {"BLEU-1":0,"BLEU-2":0,"BLEU-4":0}
    smooth = SmoothingFunction().method1
    refs_tok = [[tokenize(r)] for r in ref_list]
    hyps_tok = [tokenize(h) for h in hyp_list]
    pairs = [(r, h) for r, h in zip(refs_tok, hyps_tok) if r[0] and h]
    if not pairs:
        return {"BLEU-1":0,"BLEU-2":0,"BLEU-4":0}
    refs_tok2, hyps_tok2 = zip(*pairs)
    return {
        "BLEU-1": round(corpus_bleu(list(refs_tok2), list(hyps_tok2), weights=(1,0,0,0)), 4),
        "BLEU-2": round(corpus_bleu(list(refs_tok2), list(hyps_tok2), weights=(0.5,0.5,0,0)), 4),
        "BLEU-4": round(corpus_bleu(list(refs_tok2), list(hyps_tok2), weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth), 4),
    }

# ── ROUGE-L ───────────────────────────────────────────────────────────────────
def compute_rouge(hyp_list, ref_list):
    if not HAS_ROUGE:
        return {"ROUGE-L":0}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for h, r in zip(hyp_list, ref_list):
        if h and r:
            scores.append(scorer.score(r, h)["rougeL"].fmeasure)
    return {"ROUGE-L": round(sum(scores)/max(len(scores),1), 4)}

# ── Classification metrics ────────────────────────────────────────────────────
def compute_clf(gt_labels, pred_labels, label):
    if not HAS_SK or not gt_labels:
        return {}
    # Use all classes present in GT (dynamic, not hardcoded)
    classes = sorted(set(gt_labels))
    acc = accuracy_score(gt_labels, pred_labels)
    p, r, f, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, labels=classes, average="macro", zero_division=0)
    out = {"label": label, "N": len(gt_labels),
           "Accuracy": round(acc,4), "Precision": round(p,4),
           "Recall": round(r,4), "F1": round(f,4)}
    # per-class F1
    p2, r2, f2, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, labels=classes, average=None, zero_division=0)
    for i, cls in enumerate(classes):
        short = cls[:4]
        out[f"P_{short}"] = round(p2[i],4)
        out[f"R_{short}"] = round(r2[i],4)
        out[f"F1_{short}"] = round(f2[i],4)
    return out

# ── Main per-rider ─────────────────────────────────────────────────────────────
def run_rider(rider_key):
    cfg    = RIDER_PATHS[rider_key]
    gt_map = load_dataset_gt(cfg["dataset"])   # event_id -> {action, notes}
    data   = {m: load_jsonl(cfg[m]) for m in MODELS}

    if not gt_map:
        print(f"  [{rider_key}] No gold dataset found, skipping")
        return [], []

    # Use events present in GT and at least one model
    all_ids = sorted(gt_map.keys())
    print(f"\n{'='*70}")
    print(f"{cfg['name']} ({rider_key}) — GT events: {len(all_ids)}")
    print(f"  GT source: {cfg['dataset']}")
    print(f"{'='*70}")

    clf_rows = []
    nlp_rows = []

    for model in MODELS:
        md = data[model]
        ids = [i for i in all_ids if i in md]
        if not ids:
            print(f"\n  [{model}] No results found, skipping")
            continue

        # --- Evasive Action Classification (vs gold_candidates GT) ---
        gt_actions   = [normalize_action(gt_map[i]["action"]) for i in ids]
        pred_actions = [normalize_action(md[i].get("evasive_action","")) for i in ids]

        clf = compute_clf(gt_actions, pred_actions, f"{rider_key}_{model}")
        if clf:
            clf_rows.append({"Rider":rider_key,"Model":model,
                             **{k:v for k,v in clf.items() if k!="label"}})
            print(f"\n  [{model}] Evasive Action Classification (N={clf['N']}, GT=gold_candidates):")
            print(f"    Accuracy={clf['Accuracy']:.3f}  Precision={clf['Precision']:.3f}  Recall={clf['Recall']:.3f}  F1={clf['F1']:.3f}")

        # --- Text Generation vs gold notes ---
        ref_texts = [gt_map[i].get("notes","") or "" for i in ids]
        hyp_texts = [md[i].get("evasive_reasoning","") or "" for i in ids]
        valid = [(r,h) for r,h in zip(ref_texts,hyp_texts) if r.strip() and h.strip()]

        if valid:
            refs_v, hyps_v = zip(*valid)
            bleu  = compute_bleu(list(hyps_v), list(refs_v))
            rouge = compute_rouge(list(hyps_v), list(refs_v))
            cider = compute_cider(list(hyps_v), list(refs_v))
        else:
            bleu  = {"BLEU-1":0,"BLEU-2":0,"BLEU-4":0}
            rouge = {"ROUGE-L":0}
            cider = 0.0

        nlp_row = {"Rider":rider_key,"Model":model,"N":len(ids),"N_text":len(valid),
                   **bleu,**rouge,"CIDEr":cider}
        nlp_rows.append(nlp_row)
        print(f"  [{model}] Text gen vs gold notes ({len(valid)} pairs):")
        print(f"    BLEU-1={bleu['BLEU-1']:.3f}  BLEU-2={bleu['BLEU-2']:.3f}  BLEU-4={bleu['BLEU-4']:.3f}  ROUGE-L={rouge.get('ROUGE-L',0):.3f}  CIDEr={cider:.3f}")

    # save per-rider
    os.makedirs(cfg["out"], exist_ok=True)
    def safe_write(path, rows):
        try:
            with open(path,"w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
        except PermissionError:
            alt = path.replace(".csv","_new.csv")
            with open(alt,"w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"  [WARN] locked, wrote to {alt}")

    if clf_rows:
        out_clf = f"{cfg['out']}/{rider_key}_clf_metrics.csv"
        safe_write(out_clf, clf_rows)
        print(f"\n  Saved clf -> {out_clf}")
    if nlp_rows:
        out_nlp = f"{cfg['out']}/{rider_key}_nlp_metrics.csv"
        safe_write(out_nlp, nlp_rows)
        print(f"  Saved nlp -> {out_nlp}")

    return clf_rows, nlp_rows

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rider", default="ALL")
    args = parser.parse_args()
    keys = list(RIDER_PATHS.keys()) if args.rider.upper()=="ALL" else [args.rider.upper()]

    all_clf, all_nlp = [], []
    for k in keys:
        c, n = run_rider(k)
        all_clf.extend(c); all_nlp.extend(n)

    # combined
    os.makedirs(f"{ROOT}/metrics", exist_ok=True)
    if all_clf:
        path = f"{ROOT}/metrics/all_riders_clf_metrics.csv"
        with open(path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(all_clf[0].keys())); w.writeheader(); w.writerows(all_clf)
        print(f"\nClassification combined -> {path}")

    if all_nlp:
        path = f"{ROOT}/metrics/all_riders_nlp_metrics.csv"
        with open(path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(all_nlp[0].keys())); w.writeheader(); w.writerows(all_nlp)
        print(f"NLP combined -> {path}")

    # print final table
    if all_nlp:
        print(f"\n{'Rider':<6} {'Model':<5} {'N':>4} {'BLEU-1':>7} {'BLEU-2':>7} {'BLEU-4':>7} {'ROUGE-L':>8} {'CIDEr':>7}")
        print("-"*55)
        for r in all_nlp:
            print(f"{r['Rider']:<6} {r['Model']:<5} {r['N_text']:>4} {r['BLEU-1']:>7.3f} {r['BLEU-2']:>7.3f} {r['BLEU-4']:>7.3f} {r['ROUGE-L']:>8.3f} {r['CIDEr']:>7.3f}")

    if all_clf:
        print(f"\n{'Rider':<6} {'Model':<5} {'IRC':<10} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print("-"*50)
        for r in all_clf:
            if r.get("IRC")=="Combined":
                print(f"{r['Rider']:<6} {r['Model']:<5} {r['IRC']:<10} {r['Accuracy']:>6.3f} {r['Precision']:>6.3f} {r['Recall']:>6.3f} {r['F1']:>6.3f}")

    print("\nDone.")
