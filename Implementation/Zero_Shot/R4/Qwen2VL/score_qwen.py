import json, pandas as pd

df = pd.read_csv('gold_candidates.csv')
files = {
    'video_only': 'predictions/motor__qwen25-3b__video_only.json',
    'raw_telemetry': 'predictions/motor__qwen25-3b__raw_telemetry.json',
    'summarized_telemetry': 'predictions/motor__qwen25-3b__summarized_telemetry.json',
}
rows=[]
for cond, path in files.items():
    d = json.load(open(path))
    tp=fp=fn=tn=0
    for i, rec in enumerate(d):
        gold = df.iloc[i]['decision']
        if gold not in ('confirm','reject'):
            continue
        pred_pos = rec.get('pred_is_evasive', False)
        gold_pos = (gold == 'confirm')
        if pred_pos and gold_pos: tp+=1
        elif pred_pos and not gold_pos: fp+=1
        elif not pred_pos and gold_pos: fn+=1
        else: tn+=1
    n = tp+fp+fn+tn
    acc = (tp+tn)/n if n else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec_ = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec_/(prec+rec_) if (prec+rec_) else 0
    rows.append((cond, tp, fp, fn, tn, acc, prec, rec_, f1))
    print(cond, 'TP',tp,'FP',fp,'FN',fn,'TN',tn, 'acc=%.3f'%acc,'prec=%.3f'%prec,'rec=%.3f'%rec_,'f1=%.3f'%f1)
