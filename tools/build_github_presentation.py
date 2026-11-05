"""Build and check GitHub presentation assets using Python's standard library.

Reads saved evidence only; never trains models or rewrites Phase 3 outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/phase3"
OUT = ROOT / "docs/github"
TESTS = ("iid_test", "ood_template_test", "ood_size_test")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def svg(title, subtitle, body, height=500):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(subtitle)}</desc>
<rect width="1200" height="{height}" fill="#f8fafc"/>
<style>text{{font-family:Arial,sans-serif;fill:#16324f}} .title{{font-size:30px;font-weight:bold}} .sub{{font-size:17px;fill:#526578}} .label{{font-size:18px}} .small{{font-size:15px}}</style>
<text x="45" y="52" class="title">{html.escape(title)}</text><text x="45" y="85" class="sub">{html.escape(subtitle)}</text>{body}</svg>'''


def architecture():
    labels = [("Incidence stars", "B_x and B_y", "Rows: incident hyperedges"),
              ("Learned pair encoder", "Symmetric column features", "Shared neural networks"),
              ("Learned cost + Sinkhorn", "C_theta → coupling P", "Total transport mass = 1"),
              ("Binary classifier", "Matched features + summaries", "Class 0 or class 1")]
    body = ''
    for i, (a, b, c) in enumerate(labels):
        x = 40 + i * 295
        body += f'<rect x="{x}" y="155" width="260" height="150" rx="14" fill="white" stroke="#bacbd9"/>'
        for j, text in enumerate((a, b, c)):
            body += f'<text x="{x+16}" y="{195+j*35}" class="{"label" if j==0 else "small"}">{html.escape(text)}</text>'
        if i < 3:
            body += f'<path d="M {x+265} 230 h 22 l -6 -6 m 6 6 l -6 6" fill="none" stroke="#137f84" stroke-width="3"/>'
    body += '<text x="45" y="365" class="label">Random initialization → train-only pretraining → supervised training → validation-selected checkpoint</text>'
    body += '<text x="45" y="405" class="sub">The primary model does not consume a fixed L1 cost or the exact oracle output.</text>'
    return svg("Learning local hypergraph equivalence", "LearnedSinkhornMatcher · architecture overview · schematic, not a trained output", body, 450)


def chart(aggregate):
    names = ["LearnedSinkhornMatcher", "MLP_Baseline", "NestedDeepSets", "BipartiteEncoder", "LearnedNoSinkhorn", "LearnedNoColumnContext"]
    colors = ["#137f84", "#547fc1", "#d48532"]
    body = ''
    for j, (split, color) in enumerate(zip(TESTS, colors)):
        body += f'<rect x="{350+j*255}" y="115" width="16" height="16" fill="{color}"/><text x="{373+j*255}" y="130" class="small">{split}</text>'
    for v in (0, .25, .5, .75, 1):
        x = 340 + 760 * v
        body += f'<path d="M{x} 155 V665" stroke="#d8e1e9"/><text x="{x-12}" y="694" class="small">{v:.2f}</text>'
    for i, name in enumerate(names):
        y = 160+i*83
        body += f'<text x="35" y="{y+30}" class="small">{name}</text>'
        for j, split in enumerate(TESTS):
            row = next(r for r in aggregate if r["model"] == name and r["split"] == split)
            mean, std = float(row["accuracy_mean"]), float(row["accuracy_std"])
            yy = y+j*20
            body += f'<rect x="340" y="{yy}" width="{760*mean:.3f}" height="13" fill="{colors[j]}"/>'
            lo, hi = 340+760*max(0,mean-std), 340+760*min(1,mean+std)
            body += f'<path d="M{lo} {yy+6.5} H{hi} M{lo} {yy+2} v9 M{hi} {yy+2} v9" stroke="#16324f"/>'
            body += f'<text x="1110" y="{yy+12}" class="small">{mean*100:.2f}%</text>'
    body += '<text x="45" y="738" class="sub">Accuracy · mean ± sample SD across five seeds · error bars clipped to [0, 1]</text>'
    body += '<text x="45" y="768" class="sub">Exact and fixed-cost comparators are listed separately in the full results table.</text>'
    return svg("Generalization to larger incidence stars", "Saved final-v2 predictions independently checked against the metric CSV files", body, 800)


def examples():
    data = read_json(ROOT / "data/splits/iid_test.json")["data"]
    first = min((s for s in data if s["label"] == 0), key=lambda s:len(s["B_x"])*len(s["B_x"][0]))
    pair = [s for s in data if s["matched_pair_id"] == first["matched_pair_id"]]
    body = ''
    for p, sample in enumerate(sorted(pair, key=lambda s:s["label"], reverse=True)):
        x0 = 45 + p*595
        label = sample["label"]
        body += f'<text x="{x0}" y="135" class="label">Class {label}: {"equal row multisets" if label else "different row multisets"}</text>'
        for side, key in enumerate(("B_x", "B_y")):
            matrix = sample[key]
            cell = min(27, 210/max(len(matrix),len(matrix[0])))
            x = x0+side*265
            body += f'<text x="{x}" y="170" class="small">{key}</text>'
            for i,row in enumerate(matrix):
                for j,value in enumerate(row):
                    body += f'<rect x="{x+j*cell}" y="{190+i*cell}" width="{cell-2}" height="{cell-2}" fill="{"#137f84" if value else "#dce5ed"}"/>'
        body += f'<text x="{x0}" y="455" class="small">{html.escape(sample["sample_id"])}</text>'
    body += '<text x="45" y="500" class="sub">Actual locked IID samples · teal = 1 · gray = 0 · compare row multisets with shared columns.</text>'
    return svg("Same margins can hide different local structure", "Rows are incident hyperedges; columns are common neighboring vertices.", body, 535)


def verify_evidence():
    spec = read_json(ROOT / "phase3_spec.json")
    per_seed = rows(RESULT / "per_seed_metrics.csv")
    aggregate = rows(RESULT / "aggregate_metrics.csv")
    failures = []
    max_errors = {key: 0.0 for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")}
    total = 0
    checkpoints = set()
    splits = {sp: {s['sample_id']:s for s in read_json(ROOT / f'data/splits/{sp}.json')['data']} for sp in spec['splits']}
    hashes = {}
    for row in per_seed:
        path = ROOT / row["prediction_path"]
        if sha(path) != row["prediction_sha256"]:
            failures.append(f"Prediction hash: {path.relative_to(ROOT)}")
        ck = RESULT / f'checkpoints/{row["model"]}_seed{row["seed"]}.pt'
        checkpoints.add(ck)
        if ck not in hashes:
            hashes[ck] = sha(ck)
        if hashes[ck] != row["checkpoint_sha256"]:
            failures.append(f"Checkpoint hash: {ck.relative_to(ROOT)}")
        prediction = rows(path)
        expected = splits[row["split"]]
        ids = [p["sample_id"] for p in prediction]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            failures.append(f"Sample IDs: {path.name}")
        confusion = [[0,0],[0,0]]
        scores = []
        for p in prediction:
            label, prob, predicted = int(p["label"]), float(p["probability"]), int(p["prediction"])
            s = expected[p["sample_id"]]
            if label != s["label"] or not math.isfinite(prob) or not 0 <= prob <= 1 or predicted != int(prob >= .5):
                failures.append(f"Invalid prediction: {path.name}:{p['sample_id']}")
            if (int(p["m"]),int(p["k"])) != (len(s["B_x"]),len(s["B_x"][0])):
                failures.append(f"Dimensions: {path.name}:{p['sample_id']}")
            confusion[label][predicted] += 1
            scores.append((prob,label))
        n = len(prediction)
        total += n
        recall, f1 = [], []
        for c in (0,1):
            tp, fn, fp = confusion[c][c], confusion[c][1-c], confusion[1-c][c]
            recall.append(tp/(tp+fn))
            f1.append(2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else 0)
            if sum(confusion[c]) != int(row[f"class_{c}_count"]):
                failures.append(f"Class count: {path.name}")
        # Mann-Whitney AUC with average ranks for tied probabilities.
        scores.sort()
        rank_sum = 0.0
        start = 0
        while start < n:
            stop = start+1
            while stop < n and scores[stop][0] == scores[start][0]:
                stop += 1
            rank_sum += ((start+1+stop)/2)*sum(y for _,y in scores[start:stop])
            start = stop
        positives = sum(confusion[1])
        negatives = n-positives
        values = {"accuracy":(confusion[0][0]+confusion[1][1])/n,
                  "balanced_accuracy":statistics.mean(recall), "macro_f1":statistics.mean(f1),
                  "roc_auc":(rank_sum-positives*(positives+1)/2)/(positives*negatives)}
        for key,value in values.items():
            error = abs(value-float(row[key]))
            max_errors[key] = max(max_errors[key],error)
            if error > 1e-10:
                failures.append(f"{path.name}:{key}:{error}")
        if n != int(row["sample_count"]):
            failures.append(f"Sample count: {path.name}")
    metric_names = [x[:-5] for x in aggregate[0] if x.endswith("_mean")]
    for row in aggregate:
        group = [p for p in per_seed if p["model"] == row["model"] and p["split"] == row["split"]]
        if sorted(int(p["seed"]) for p in group) != sorted(spec["seeds"]) or int(row["seed_count"]) != len(group):
            failures.append(f"Seed coverage: {row['model']}:{row['split']}")
        for key in metric_names:
            numbers = [float(p[key]) for p in group]
            for suffix,value in (("mean",statistics.mean(numbers)),("std",statistics.stdev(numbers))):
                if abs(value-float(row[f"{key}_{suffix}"])) > 1e-10:
                    failures.append(f"Aggregate: {row['model']}:{row['split']}:{key}:{suffix}")
    expected_keys = {(m,str(s),sp) for m in spec["model_roles"] for s in spec["seeds"] for sp in spec["splits"]}
    actual_keys = [(p["model"],p["seed"],p["split"]) for p in per_seed]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        failures.append("Model/seed/split coverage")
    manifest = read_json(RESULT / "final_artifact_manifest.json")
    drift = []
    for entry in manifest["entries"]:
        path = ROOT / entry["path"]
        if not path.is_file() or sha(path) != entry["sha256"]:
            drift.append(entry["path"])
    # Document pre-existing handoff drift, but never conceal scientific file drift.
    unexpected = [p for p in drift if p != "CODEX_PHASE3_STATUS.md"]
    failures.extend(f"Historical manifest drift: {p}" for p in unexpected)
    lock = read_json(RESULT / "learned_benchmark_lock.json")
    for split,digest in lock["split_sha256"].items():
        if sha(ROOT / f"data/splits/{split}.json") != digest:
            failures.append(f"Locked split hash: {split}")
    return {"scope":"Saved prediction, aggregate, checkpoint and historical-manifest verification; no retraining or numerical-audit rerun",
            "status":"PASS" if not failures else "FAIL", "prediction_files":len(per_seed),
            "prediction_rows":total,"checkpoints":len(checkpoints),"aggregate_rows":len(aggregate),
            "recomputed_prediction_metrics":list(max_errors),"maximum_prediction_metric_errors":max_errors,
            "aggregate_metrics_recomputed":metric_names,"historical_manifest_entries_checked":len(manifest["entries"]),
            "historical_manifest_drift":drift,"failures":failures}


def outputs():
    aggregate = rows(RESULT / "aggregate_metrics.csv")
    table = ["# Final v2 results", "", "Saved results, mean ± sample standard deviation across five seeds (42, 100, 2026, 777, 999).", "", "| Model | Role | Split | n per seed | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC | Loss |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        if r["split"] not in TESTS:
            continue
        vals = [f'{float(r[k+"_mean"]):.6f} ± {float(r[k+"_std"]):.6f}' for k in ("accuracy","balanced_accuracy","macro_f1","roc_auc","loss")]
        table.append(f'| `{r["model"]}` | `{r["role"]}` | `{r["split"]}` | {int(float(r["sample_count_mean"]))} | '+" | ".join(vals)+" |")
    table += ["", "Source: [aggregate_metrics.csv](../../results/phase3/aggregate_metrics.csv).", "", "The primary is specification-designated, not chosen from the test ranking. It is the best learned model by mean macro F1 over these three test splits. Four algorithmic/oracle comparators tie its perfect classification scores. The generated presentation check recomputes accuracy, balanced accuracy, macro F1 and ROC-AUC from all saved predictions, recomputes every aggregate column from per-seed metrics, and verifies checkpoint and prediction hashes. Loss is aggregated from saved per-seed values; it is not re-evaluated from checkpoint inference in this documentation pass."]
    return {OUT/"architecture.svg":architecture(), OUT/"generalization.svg":chart(aggregate),
            OUT/"incidence_examples.svg":examples(), OUT/"RESULTS.md":"\n".join(table)+"\n"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    mismatches = []
    generated = outputs()
    for path,content in generated.items():
        if path.suffix == ".svg":
            ET.fromstring(content)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                mismatches.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(content,encoding="utf-8",newline="\n")
    result = verify_evidence()
    result["generated_asset_mismatches"] = mismatches
    if mismatches:
        result["status"] = "FAIL"
    evidence = OUT/"presentation_validation.json"
    if not args.check:
        evidence.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    elif not evidence.is_file() or read_json(evidence) != result:
        result["status"] = "FAIL"
        result["failures"].append("Saved presentation validation differs; rebuild required")
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
