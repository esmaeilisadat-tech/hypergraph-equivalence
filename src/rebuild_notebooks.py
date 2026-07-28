"""Build the two required artifact-backed Phase 3 notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from src.phase3_data import PROJECT_ROOT

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def _code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def _markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


SETUP = r"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "phase3_spec.json").exists():
    ROOT = ROOT.parent
RESULTS = ROOT / "results" / "phase3"

def read_json(relative):
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))

spec = json.loads((ROOT / "phase3_spec.json").read_text(encoding="utf-8"))
aggregate = pd.read_csv(RESULTS / "aggregate_metrics.csv")
per_seed = pd.read_csv(RESULTS / "per_seed_metrics.csv")
zero_shot = pd.read_csv(RESULTS / "zero_shot_metrics.csv")
label_signal = pd.read_csv(
    RESULTS / "controls" / "supervised_only_label_signal_control_metrics.csv"
)
pretraining_confound = pd.read_csv(
    RESULTS / "controls" / "full_pipeline_pretraining_confound_metrics.csv"
)
split_audit = read_json("split_overlap_audit.json")
benchmark_audit = read_json("learned_benchmark_audit.json")
benchmark_lock = read_json("learned_benchmark_lock.json")
gate_a = read_json("gate_a_report.json")
gate_b = read_json("gate_b_report.json")
gate_c = read_json("gate_c_report.json")
numerical = read_json("numerical_audit.json")
scientific = read_json("scientific_claim_audit.json")
selection = read_json("model_selection/model_selection_protocol.json")

assert spec["primary_model"] == "LearnedSinkhornMatcher"
print("Loaded final benchmark:", spec["benchmark_version"])
print("Designated primary:", spec["primary_model"])
"""

DATA_AUDIT = r"""
rows = []
for split, report in benchmark_audit["split_reports"].items():
    rows.append({
        "split": split,
        "samples": report["sample_count"],
        "class 0": report["class_counts"]["0"],
        "class 1": report["class_counts"]["1"],
        "negative cost min": report["negative_hungarian_cost"]["minimum"],
        "negative cost max": report["negative_hungarian_cost"]["maximum"],
        "hard negatives (cost 4)": report[
            "negative_hungarian_cost"
        ]["hard_negative_count_cost_4"],
        "max nuisance AUC": report["nuisance_scalar_max_auc"],
    })
display(pd.DataFrame(rows))
assert split_audit["status"] == "PASS"
assert benchmark_audit["status"] == "PASS"
assert split_audit["problem_count"] == 0
assert benchmark_lock["locked_before_first_final_test_evaluation"] is True
assert benchmark_lock["locked_before_first_optimization"] is False
"""

LEARNING_EVIDENCE = r"""
primary = spec["primary_model"]
primary_zero = zero_shot[
    (zero_shot["model"] == primary)
    & zero_shot["split"].isin(["train", "validation"])
]
primary_trained = per_seed[
    (per_seed["model"] == primary)
    & per_seed["split"].isin(["train", "validation"])
]
learning = (
    primary_zero.groupby("split")[["accuracy", "macro_f1", "roc_auc"]].mean()
    .add_prefix("zero_shot_")
    .join(
        primary_trained.groupby("split")[
            ["accuracy", "macro_f1", "roc_auc"]
        ].mean().add_prefix("trained_")
    )
)
display(learning)
assert len(primary_trained["seed"].unique()) == 5
assert scientific["checks"]["five_independent_primary_seeds"]["status"] == "PASS"
assert scientific["checks"]["training_improves_primary"]["status"] == "PASS"
"""

METRICS = r"""
test_splits = ["iid_test", "ood_template_test", "ood_size_test"]
columns = [
    "model", "role", "split", "loss_mean", "accuracy_mean", "accuracy_std",
    "balanced_accuracy_mean", "macro_f1_mean", "macro_f1_std",
    "roc_auc_mean", "roc_auc_std", "seed_count",
]
test_metrics = aggregate[aggregate["split"].isin(test_splits)][columns]
role_order = {
    "primary_learned": 0, "baseline": 1, "ablation": 2,
    "algorithmic_hybrid": 3, "algorithmic_baseline": 4, "oracle": 5,
}
test_metrics = test_metrics.assign(
    role_order=test_metrics["role"].map(role_order)
).sort_values(["role_order", "model", "split"]).drop(columns="role_order")
display(test_metrics)
assert len(test_metrics) == 10 * 3
assert (test_metrics["seed_count"] == 5).all()
"""

PLOT = r"""
learned_roles = ["primary_learned", "baseline", "ablation"]
plot_data = test_metrics[
    test_metrics["role"].isin(learned_roles)
].pivot(index="model", columns="split", values="macro_f1_mean")
ax = plot_data.plot(kind="bar", figsize=(12, 5), ylim=(0, 1.03))
ax.set_title("Learned-model macro F1 on locked final-v2 test splits")
ax.set_ylabel("Mean macro F1 across five seeds")
ax.set_xlabel("")
ax.legend(title="Split")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
"""

CONTROLS = r"""
control_summary = (
    label_signal[
        label_signal["split"].isin(
            ["iid_test", "ood_template_test", "ood_size_test"]
        )
    ]
    .groupby(["control", "split"])
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        seeds=("seed", "nunique"),
    )
    .reset_index()
)
display(control_summary)
display(
    pretraining_confound[
        pretraining_confound["split"].isin(
            ["iid_test", "ood_template_test", "ood_size_test"]
        )
    ]
    .groupby(["control", "split"])
    .agg(accuracy_mean=("accuracy", "mean"))
    .reset_index()
)
display(pd.DataFrame([
    {
        "check": name,
        "status": evidence["status"],
    }
    for name, evidence in scientific["checks"].items()
]))
assert scientific["status"] == "PASS"
assert (control_summary["seeds"] == 5).all()
"""

GATES = r"""
gate_rows = [
    {
        "gate": "A",
        "status": gate_a["status"],
        "completed": gate_a.get("oracle_sample_count"),
        "key value": gate_a.get("oracle_accuracy"),
    },
    {
        "gate": "B",
        "status": gate_b["status"],
        "completed": gate_b["completed_seed_count"],
        "key value": min(row["accuracy"] for row in gate_b["per_seed"]),
    },
    {
        "gate": "C",
        "status": gate_c["status"],
        "completed": gate_c["completed_seed_count"],
        "key value": min(
            row["train_metrics"]["accuracy"] for row in gate_c["per_seed"]
        ),
    },
]
display(pd.DataFrame(gate_rows))
assert gate_a["status"] == gate_b["status"] == gate_c["status"] == "PASS"
assert gate_b["completed_seed_count"] == gate_c["completed_seed_count"] == 5
"""

NUMERICAL = r"""
quantities = pd.DataFrame([
    {
        "quantity": name,
        "maximum": values["maximum"],
        "p95": values["p95"],
        "p99": values["p99"],
        "tolerance": values["tolerance"],
        "failures": values["failed_count"],
    }
    for name, values in numerical["quantities"].items()
])
display(quantities)
print("Audited seed-sample cases:", numerical["audited_sample_count"])
print("Failed cases:", numerical["failed_sample_count"])
assert numerical["status"] == "PASS"
assert numerical["failed_sample_count"] == 0
assert (quantities["failures"] == 0).all()
"""

REPRODUCTION = r"""
commands = [
    "python -m src.audit_learning_benchmark",
    "python -m src.audit_splits",
    "python -m src.run_final_experiments",
    "python -m src.run_gates",
    "python -m src.run_label_signal_controls",
    "python -m src.scientific_claim_audit",
    "python -m src.audit_model_numerics",
    "python -m src.reporting",
    "python -m pytest -q",
]
print("\n".join(commands))
"""


def research_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        _markdown(
            """# Phase 3 Research — Learned Hypergraph Equivalence

This clean-executed notebook reads final artifacts rather than recomputing or
silently altering them. It separates the designated learned primary from
fixed-cost algorithmic controls and the exact oracle."""
        ),
        _code(SETUP),
        _markdown(
            """## Scope, split definitions, and benchmark difficulty

Class 0 is EE-equivalent but not strongly structurally equivalent; class 1 has
the same multiset of incident neighbor sets. Each matched pair shares nuisance
statistics. Train and validation are byte-identical to preserved development
v1; final-v2 IID, OOD-template, and OOD-size samples use new, unseen offsets.
The selected v5 recipe was frozen before these final-v2 tests were evaluated.
This is a post-selection final-test lock—not a claim that v2 existed before
development optimization."""
        ),
        _code(DATA_AUDIT),
        _markdown(
            """## Model identity and methodology

`LearnedSinkhornMatcher` is the designated primary. It starts from random
weights, encodes column pairs symmetrically, learns row-pair scores/costs,
computes normalized-mass Sinkhorn transport, and learns the final classifier.
It receives only the two incidence matrices. It does not receive fixed L1
cost, Hungarian output, labels, IDs, hashes, or split names.

`MLP_Baseline`, `NestedDeepSets`, and `BipartiteEncoder` are learned baselines;
`LearnedNoSinkhorn` and `LearnedNoColumnContext` are ablations.
`DeterministicDifferentiableMatching`, `FixedL1ResidualSinkhorn`,
`FixedL1StraightThroughHungarian`, and `ExactHungarianOracle` are reported
separately as algorithmic/hybrid/oracle evidence."""
        ),
        _markdown(
            """## Zero-shot versus trained evidence

Chance-level zero-shot primary behavior, five distinct initial/final states,
five seed-dependent training orders, and a material post-training improvement
are mandatory checks. Candidate v1 exposed development-test metrics; those v1
tests were retired from final claims. Candidate v5 was selected using only
train/validation evidence, then evaluated on unseen final-v2 tests."""
        ),
        _code(LEARNING_EVIDENCE),
        _markdown(
            """## Five-seed IID and OOD results

Means and sample standard deviations use locked seeds 42, 100, 2026, 777, and
999. Validation alone selects checkpoints; all test splits are evaluation-only."""
        ),
        _code(METRICS),
        _code(PLOT),
        _markdown(
            """## Negative and positive controls

The supervised-label-signal shuffled/random controls disable only label-free
structural pretraining, then test whether corrupted class labels can create
held-out success. The original full-pipeline invalid-label run is also shown:
it is a preserved failed/confounded control because self-supervised structural
pretraining supplies task signal without class labels. The label-access
positive control must score perfectly, while source inspection must show that
the real primary cannot access labels or metadata. Exact Hungarian establishes
that the task is algorithmically separable without being misreported as
learned discovery."""
        ),
        _code(CONTROLS),
        _markdown(
            """## Gates and normalized Sinkhorn mass

Gate A verifies exact labels and convergence. A coupling is normalized to
total mass 1; with `m` equally weighted valid rows, a hard matched entry is
approximately `1/m`, not 1. Gate B retains the accuracy and macro-F1 threshold
of 0.99 and per-class F1 threshold of 0.98 for all seeds. Gate C uses
validation-only selection and checks the unchanged training thresholds."""
        ),
        _code(GATES),
        _markdown(
            """## Complete float64 structural audit

Every primary checkpoint is tested on every final-v2 IID, OOD-template, and
OOD-size sample using compact unpadded float64 matrices and 500 Sinkhorn
iterations. Checks cover transpose equivariance, matching/equality/final-score
invariance, mass, marginals, finiteness, non-negativity, assignment transpose,
row permutations, and joint column permutations."""
        ),
        _code(NUMERICAL),
        _markdown(
            """## Limitations and interpretation

The benchmark is synthetic and targets the local incidence-star definition;
it does not establish universal empirical-hypergraph performance. Five seeds
measure optimization variability on fixed data, not population uncertainty.
The final-v2 test remedy prevents the disclosed v1 incident from supporting
final claims, but the chronology remains a limitation that must be reported.
Exact fixed-cost success is useful algorithmic evidence, not learned evidence.
The preserved full-pipeline invalid-label FAIL demonstrates that the
self-supervised pretraining can learn structural task signal without class
labels; the separate supervised-only control supports only the narrower
claim about label corruption."""
        ),
        _markdown("## Reproducibility commands"),
        _code(REPRODUCTION),
    ]
    return notebook


def persian_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        _markdown(
            """# توضیح مرحلهٔ سوم: یادگیری هم‌ارزی ابرگراف

این دفترچه فقط خروجی‌های نهایی و راستی‌آزمایی‌شده را می‌خواند. مدل اصلی
`LearnedSinkhornMatcher` است و نتایج آن از حل‌گرهای ثابت و Oracle جدا گزارش
می‌شود."""
        ),
        _code(SETUP),
        _markdown(
            """## داده‌ها، قفل آزمون و سختی مسئله

در کلاس صفر، حاشیه‌های سطر و ستون حفظ می‌شوند ولی چندمجموعهٔ همسایه‌ها برابر
نیست؛ در کلاس یک این چندمجموعه دقیقاً برابر است. آموزش و اعتبارسنجی همان
داده‌های توسعهٔ v1 هستند، اما سه آزمون نهایی v2 با offsetهای تازه ساخته
شده‌اند. مدل v5 فقط با آموزش و اعتبارسنجی انتخاب شد و سپس تعریف مدل و دستور
آموزش پیش از اولین ارزیابی آزمون v2 قفل شد. بنابراین ادعای درست «قفل پیش از
آزمون نهایی» است، نه «قفل پیش از همهٔ بهینه‌سازی‌های توسعه»."""
        ),
        _code(DATA_AUDIT),
        _markdown(
            """## هویت مدل و روش

مدل اصلی از وزن‌های تصادفی شروع می‌کند، نمایش متقارن جفت‌ستون‌ها و هزینهٔ
جفت‌سطرها را یاد می‌گیرد، انتقال Sinkhorn با جرم کل یک را می‌سازد و طبقه‌بند
نهایی را آموزش می‌دهد. ورودی واقعی مدل فقط دو ماتریس وقوع است؛ هزینهٔ L1
ثابت، نتیجهٔ Hungarian، برچسب، شناسه و هش به آن داده نمی‌شود.

مدل‌های دارای هزینهٔ ثابت و `ExactHungarianOracle` کنترل‌های الگوریتمی‌اند و
نباید به‌عنوان کشف یادگرفته‌شده تفسیر شوند."""
        ),
        _markdown(
            """## مقایسهٔ پیش از آموزش و پس از آموزش

برای ادعای یادگیری، عملکرد مدل اصلی پیش از آموزش باید نزدیک شانس باشد؛ سپس
پنج مقداردهی اولیه، پنج ترتیب داده و پنج checkpoint نهایی مستقل بررسی
می‌شوند. اجرای ناموفق v1 بخشی از معیارهای آزمون توسعه را آشکار کرد؛ به همین
علت آزمون‌های v1 از ادعاهای نهایی کنار گذاشته شدند و آزمون‌های تازهٔ v2
جایگزین شدند."""
        ),
        _code(LEARNING_EVIDENCE),
        _markdown(
            """## نتایج پنج seed روی IID و OOD

جدول از seedهای 42، 100، 2026، 777 و 999 استفاده می‌کند. checkpoint فقط با
اعتبارسنجی انتخاب شده و IID، OOD-الگو و OOD-اندازه صرفاً برای ارزیابی نهایی
هستند."""
        ),
        _code(METRICS),
        _code(PLOT),
        _markdown(
            """## کنترل‌های علمی

آموزش با برچسب‌های درهم‌ریخته و تصادفی باید نشان دهد که خط لوله بدون سیگنال
معتبر نمی‌تواند موفقیت آزمون بسازد. کنترل مثبت دسترسی به برچسب باید دقیقاً
موفق باشد، اما بازرسی کد باید ثابت کند مدل واقعی به برچسب یا فراداده دسترسی
ندارد. Oracle فقط جدایی الگوریتمی مسئله را نشان می‌دهد."""
        ),
        _code(CONTROLS),
        _markdown(
            """## Gateها و جرم Sinkhorn

جرم کل coupling برابر یک است. با `m` سطر معتبر هم‌وزن، جرم هر تطبیق سخت
تقریباً `1/m` است، نه یک. Gate B آستانه‌های اصلی 0.99 برای accuracy و
macro-F1 و 0.98 برای F1 هر کلاس را در هر پنج seed حفظ می‌کند."""
        ),
        _code(GATES),
        _markdown(
            """## ممیزی عددی float64

برای هر پنج checkpoint و تمام نمونه‌های سه آزمون نهایی، تقارن تعویض ریشه،
ترانهاده‌شدن هزینه و coupling، پایداری امتیاز نهایی، جرم و حاشیه‌ها، نبود
NaN/Inf، نامنفی‌بودن و ناوردایی نسبت به جایگشت سطر و ستون بررسی می‌شود."""
        ),
        _code(NUMERICAL),
        _markdown(
            """## محدودیت‌ها

داده‌ها مصنوعی‌اند و فقط تعریف محلی incidence-star و خانوادهٔ مولد قفل‌شده
را می‌سنجند. پنج seed عدم‌قطعیت بهینه‌سازی را نشان می‌دهد، نه عدم‌قطعیت جامعهٔ
واقعی. اصلاح آزمون v2 اثر رخداد افشاشدهٔ v1 را از ادعاهای نهایی جدا می‌کند،
ولی این تاریخچه باید شفاف گزارش شود. موفقیت روش‌های ثابت دلیل یادگیری نیست."""
        ),
        _markdown("## فرمان‌های بازتولید"),
        _code(REPRODUCTION),
    ]
    return notebook


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(research_notebook(), NOTEBOOK_DIR / "Phase3_Research.ipynb")
    nbf.write(persian_notebook(), NOTEBOOK_DIR / "Phase3_Explained_FA.ipynb")


if __name__ == "__main__":
    main()
