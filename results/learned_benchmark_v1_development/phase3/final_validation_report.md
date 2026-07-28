# Phase 3 Hypergraph Equivalence — Final Validation Report

## Technical summary

**Overall status: PASS.** The designated primary model is
`MatchingNetwork`; the numerically best-performing models under the locked ranking
(mean macro F1 across IID, OOD-template, and OOD-size) are `LearnedCostSinkhorn`, `MatchingNetwork`, `StraightThroughHungarian` (exact tie).
The primary result is: IID accuracy 1.000000 ± 0.000000, macro F1 1.000000 ± 0.000000; OOD-size accuracy 1.000000 ± 0.000000, macro F1 1.000000 ± 0.000000; OOD-template accuracy 1.000000 ± 0.000000, macro F1 1.000000 ± 0.000000.

The legacy Antigravity evidence was not reused as final evidence. Its leaked
train/validation/IID structures, weakened Gate B wording, inconsistent model
attribution, and incomplete numerical schema were replaced by a deterministic
zero-overlap corpus and a clean five-seed execution.

## The locked model comparison separates matching from non-matching baselines

| Model | Role | Split | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC | Seeds |
|---|---|---|---:|---:|---:|---:|---:|
| BipartiteEncoder | baseline | IID | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.466940 ± 0.039690 | 0.499986 ± 0.000011 | 5 |
| BipartiteEncoder | baseline | OOD-size | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.435789 ± 0.078187 | 0.499980 ± 0.000024 | 5 |
| BipartiteEncoder | baseline | OOD-template | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.468557 ± 0.042889 | 0.500005 ± 0.000015 | 5 |
| LearnedCostSinkhorn | ablation | IID | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| LearnedCostSinkhorn | ablation | OOD-size | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| LearnedCostSinkhorn | ablation | OOD-template | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| MLP_Baseline | baseline | IID | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.462202 ± 0.054702 | 0.500000 ± 0.000000 | 5 |
| MLP_Baseline | baseline | OOD-size | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.442061 ± 0.063182 | 0.500000 ± 0.000000 | 5 |
| MLP_Baseline | baseline | OOD-template | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.464308 ± 0.050666 | 0.500000 ± 0.000000 | 5 |
| MatchingNetwork | primary | IID | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| MatchingNetwork | primary | OOD-size | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| MatchingNetwork | primary | OOD-template | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| NestedDeepSets | baseline | IID | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.398745 ± 0.089572 | 0.499999 ± 0.000021 | 5 |
| NestedDeepSets | baseline | OOD-size | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.396077 ± 0.085934 | 0.500013 ± 0.000035 | 5 |
| NestedDeepSets | baseline | OOD-template | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.399843 ± 0.090070 | 0.500004 ± 0.000012 | 5 |
| NoSinkhorn | ablation | IID | 0.602500 ± 0.140690 | 0.602500 ± 0.140690 | 0.591346 ± 0.138172 | 0.607141 ± 0.146554 | 5 |
| NoSinkhorn | ablation | OOD-size | 0.517500 ± 0.027386 | 0.517500 ± 0.027386 | 0.369108 ± 0.055260 | 0.539145 ± 0.052896 | 5 |
| NoSinkhorn | ablation | OOD-template | 0.635500 ± 0.185751 | 0.635500 ± 0.185751 | 0.629073 ± 0.188044 | 0.636961 ± 0.187164 | 5 |
| StraightThroughHungarian | ablation | IID | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| StraightThroughHungarian | ablation | OOD-size | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| StraightThroughHungarian | ablation | OOD-template | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |

Every row uses the same locked splits and seeds 42, 100, 2026, 777, and 999.
Means and sample standard deviations are computed directly from verified
per-seed prediction CSVs. Accuracy is the fraction correctly classified at
threshold 0.5; balanced accuracy averages class recalls; macro F1 averages the
two class F1 values; ROC-AUC and PR-AUC use probabilities; Brier score is mean
squared probability error.

<!-- HEADLINE_METRICS_JSON_START -->
[{"mean": 0.5, "metric": "accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.4669399718640498, "metric": "macro_f1", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0396899236464214}, {"mean": 0.4999862499999999, "metric": "roc_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 1.1180339887493514e-05}, {"mean": 0.5000290061598058, "metric": "pr_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 3.4444276764297065e-05}, {"mean": 0.2500092446804046, "metric": "brier_score", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 1.3407789603468388e-05}, {"mean": 0.5, "metric": "accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.4357886838177308, "metric": "macro_f1", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.078187492868889}, {"mean": 0.49998, "metric": "roc_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 2.436698586201361e-05}, {"mean": 0.5001146224660102, "metric": "pr_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0001107817524824}, {"mean": 0.2500100433826446, "metric": "brier_score", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 1.8686044254174083e-05}, {"mean": 0.5, "metric": "accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.4685570535430172, "metric": "macro_f1", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0428893878104055}, {"mean": 0.500005, "metric": "roc_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 1.4757148691422894e-05}, {"mean": 0.5000518006238555, "metric": "pr_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 4.579421401213784e-05}, {"mean": 0.2500091433525085, "metric": "brier_score", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 1.301086252090549e-05}, {"mean": 1.0, "metric": "accuracy", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.0001577384391566, "metric": "brier_score", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 4.538422523560593e-08}, {"mean": 1.0, "metric": "accuracy", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.0001577705814269, "metric": "brier_score", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 5.09849369384702e-08}, {"mean": 1.0, "metric": "accuracy", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.0001577313058077, "metric": "brier_score", "model": "LearnedCostSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 4.550336124458591e-08}, {"mean": 0.5, "metric": "accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.4622016775541257, "metric": "macro_f1", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0547018431905221}, {"mean": 0.4999999999999999, "metric": "roc_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 6.206335383118183e-17}, {"mean": 0.5, "metric": "pr_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.2505246460437774, "metric": "brier_score", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0004776705942417}, {"mean": 0.5, "metric": "accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.4420609972066828, "metric": "macro_f1", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0631817683953535}, {"mean": 0.5, "metric": "roc_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 6.206335383118183e-17}, {"mean": 0.5, "metric": "pr_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.250590181350708, "metric": "brier_score", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0005732765777215}, {"mean": 0.5, "metric": "accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.4643078874639273, "metric": "macro_f1", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0506655424873205}, {"mean": 0.5, "metric": "roc_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 1.2412670766236366e-16}, {"mean": 0.5, "metric": "pr_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.2505536317825317, "metric": "brier_score", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.000550002403228}, {"mean": 1.0, "metric": "accuracy", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.0001578503724886, "metric": "brier_score", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.0001578880037413, "metric": "brier_score", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.0001578434894327, "metric": "brier_score", "model": "MatchingNetwork", "role": "primary", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.5, "metric": "accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.3987448340154565, "metric": "macro_f1", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0895722535642018}, {"mean": 0.4999993749999999, "metric": "roc_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 2.1009670392448472e-05}, {"mean": 0.5001107566186415, "metric": "pr_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0001840432425489}, {"mean": 0.250002521276474, "metric": "brier_score", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 4.241499775403096e-06}, {"mean": 0.5, "metric": "accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.3960773201157816, "metric": "macro_f1", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0859339486666261}, {"mean": 0.5000125000000001, "metric": "roc_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 3.5355339059395445e-05}, {"mean": 0.5000982438883789, "metric": "pr_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 8.512113212660575e-05}, {"mean": 0.2500024557113647, "metric": "brier_score", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 4.2340594796237465e-06}, {"mean": 0.5, "metric": "accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.399843330659488, "metric": "macro_f1", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0900704104971818}, {"mean": 0.50000375, "metric": "roc_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 1.1775902300919633e-05}, {"mean": 0.5000275468430411, "metric": "pr_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 2.567743591585348e-05}, {"mean": 0.2500025331974029, "metric": "brier_score", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 4.219897300386185e-06}, {"mean": 0.6025, "metric": "accuracy", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.1406902626339151}, {"mean": 0.6025, "metric": "balanced_accuracy", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.140690262633915}, {"mean": 0.591345600840936, "metric": "macro_f1", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.1381716291358989}, {"mean": 0.607140625, "metric": "roc_auc", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.1465541402699816}, {"mean": 0.5761131852675081, "metric": "pr_auc", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.1028212038718026}, {"mean": 0.2330055266618728, "metric": "brier_score", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0376518937076827}, {"mean": 0.5175, "metric": "accuracy", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0273861278752583}, {"mean": 0.5175, "metric": "balanced_accuracy", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0273861278752583}, {"mean": 0.3691078593068276, "metric": "macro_f1", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0552603126842948}, {"mean": 0.5391450000000001, "metric": "roc_auc", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0528961021720882}, {"mean": 0.5236219221848264, "metric": "pr_auc", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0300046115895307}, {"mean": 0.2500442147254943, "metric": "brier_score", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0001121132583159}, {"mean": 0.6355000000000001, "metric": "accuracy", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.1857514300079544}, {"mean": 0.6355000000000001, "metric": "balanced_accuracy", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.1857514300079545}, {"mean": 0.6290731944541994, "metric": "macro_f1", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.1880439118587477}, {"mean": 0.6369612499999999, "metric": "roc_auc", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.1871639162757182}, {"mean": 0.6027981674594358, "metric": "pr_auc", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.1404730290988919}, {"mean": 0.2257429420948028, "metric": "brier_score", "model": "NoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0536832663732886}, {"mean": 1.0, "metric": "accuracy", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.0001564603880979, "metric": "brier_score", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.0001565035490784, "metric": "brier_score", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.00015645143867, "metric": "brier_score", "model": "StraightThroughHungarian", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}]
<!-- HEADLINE_METRICS_JSON_END -->

## Scope and data definitions eliminate the inherited leakage

The task distinguishes class 2 pairs (EE-equivalent incidence stars that are
not strongly structurally equivalent) from class 3 pairs (equal multisets of
incident neighbor sets). Each negative/positive matched pair shares a generated
base, dimensions, margins, and nuisance statistics. Train, validation, IID,
OOD-template, and OOD-size contain 800, 800, 800, 800, and 400 samples,
respectively. OOD-template uses three degree-preserving switches; OOD-size uses
15–30 incident rows and 12–20 neighbor columns.

Split audit status is `PASS` with
`0` problems and zero prohibited sample-ID,
matched-pair, base-hash, or invariant-structure overlaps in every split pair.
Generation and loading are deterministic, labels agree with the exact multiset
oracle, and padding/masks were independently checked.

## Model and experimental design

`MatchingNetwork` is the designated primary model—not whichever model happens
to rank first. It uses fixed normalized L1 row cost, a balanced log-domain
Sinkhorn coupling, and a learned symmetric classifier. The baselines are
`MLP_Baseline`, `NestedDeepSets`, and `BipartiteEncoder`; the ablations are
`LearnedCostSinkhorn`, `StraightThroughHungarian`, and `NoSinkhorn`.

For every model and seed, only validation macro F1 followed by validation loss
selects the checkpoint. IID and both OOD splits are evaluation-only. The
artifacts preserve selection epoch, selected temperature, Sinkhorn iteration
counts, device, dtype, checkpoint hash, and per-sample probabilities.

## Gate A verifies the oracle and normalized coupling

Gate A is `PASS` on `3600` samples with
oracle accuracy `1.000000`. A normalized coupling has
total mass 1; for `m` equally weighted valid rows, each hard matched entry is
approximately `1/m`, not 1. The 500-iteration float64 evidence has maximum total
mass error `2.220e-16`, row residual
`6.148e-15`, and column residual
`5.274e-15` against tolerance
`1.0e-08`.

## Gate B passes the unchanged 0.99 rule for every seed

- Seed 42: PASS; accuracy 1.000000; macro F1 1.000000; loss 0.004119; failed IDs: none.
- Seed 100: PASS; accuracy 1.000000; macro F1 1.000000; loss 0.004119; failed IDs: none.
- Seed 2026: PASS; accuracy 1.000000; macro F1 1.000000; loss 0.004119; failed IDs: none.
- Seed 777: PASS; accuracy 1.000000; macro F1 1.000000; loss 0.004119; failed IDs: none.
- Seed 999: PASS; accuracy 1.000000; macro F1 1.000000; loss 0.004119; failed IDs: none.

The rule remains accuracy ≥ 0.99, macro F1 ≥ 0.99, and each class F1 ≥ 0.98.
No sample or threshold was removed after observing results.

## Gate C uses validation-only selection and passes every seed

- Seed 42: PASS; training accuracy 1.000000; macro F1 1.000000; selected epoch 160; temperature 0.00500000.
- Seed 100: PASS; training accuracy 1.000000; macro F1 1.000000; selected epoch 160; temperature 0.00500000.
- Seed 2026: PASS; training accuracy 1.000000; macro F1 1.000000; selected epoch 160; temperature 0.00500000.
- Seed 777: PASS; training accuracy 1.000000; macro F1 1.000000; selected epoch 160; temperature 0.00500000.
- Seed 999: PASS; training accuracy 1.000000; macro F1 1.000000; selected epoch 160; temperature 0.00500000.

The locked training thresholds are accuracy, balanced accuracy, and macro F1
at least 0.95, with each class recall at least 0.90.

## The complete float64 audit has no unexplained failures

The audit status is `PASS` over
`10000` seed–sample evaluations, using compact
unpadded submatrices, explicit uniform marginals, deterministic permutations,
float64, and `500` iterations.

- `cost_finite_violation`: maximum `0.000e+00`, p99 `0.000e+00`, failures `0`/`10000`, tolerance `1.0e-08`.
- `coupling_finite_violation`: maximum `0.000e+00`, p99 `0.000e+00`, failures `0`/`10000`, tolerance `1.0e-08`.
- `coupling_nonnegativity_violation`: maximum `0.000e+00`, p99 `0.000e+00`, failures `0`/`10000`, tolerance `1.0e-08`.
- `cost_transpose_error`: maximum `0.000e+00`, p99 `0.000e+00`, failures `0`/`10000`, tolerance `1.0e-08`.
- `coupling_transpose_error`: maximum `7.252e-15`, p99 `2.501e-16`, failures `0`/`10000`, tolerance `1.0e-08`.
- `assignment_transpose_error`: maximum `7.252e-15`, p99 `2.501e-16`, failures `0`/`10000`, tolerance `1.0e-08`.
- `matching_cost_swap_error`: maximum `2.914e-16`, p99 `1.388e-16`, failures `0`/`10000`, tolerance `1.0e-08`.
- `equality_score_swap_error`: maximum `1.110e-15`, p99 `4.441e-16`, failures `0`/`10000`, tolerance `1.0e-08`.
- `final_logit_swap_error`: maximum `0.000e+00`, p99 `0.000e+00`, failures `0`/`10000`, tolerance `1.0e-08`.
- `final_probability_swap_error`: maximum `0.000e+00`, p99 `0.000e+00`, failures `0`/`10000`, tolerance `1.0e-08`.
- `total_coupling_mass_error`: maximum `4.441e-16`, p99 `2.220e-16`, failures `0`/`10000`, tolerance `1.0e-08`.
- `row_marginal_residual`: maximum `1.017e-13`, p99 `9.529e-14`, failures `0`/`10000`, tolerance `1.0e-08`.
- `column_marginal_residual`: maximum `1.055e-13`, p99 `9.531e-14`, failures `0`/`10000`, tolerance `1.0e-08`.
- `node_row_permutation_coupling_error`: maximum `9.097e-15`, p99 `2.776e-16`, failures `0`/`10000`, tolerance `1.0e-08`.
- `node_row_permutation_logit_error`: maximum `5.902e-12`, p99 `2.842e-14`, failures `0`/`10000`, tolerance `1.0e-08`.
- `node_row_permutation_probability_error`: maximum `1.030e-13`, p99 `3.331e-16`, failures `0`/`10000`, tolerance `1.0e-08`.
- `joint_column_permutation_cost_error`: maximum `0.000e+00`, p99 `0.000e+00`, failures `0`/`10000`, tolerance `1.0e-08`.
- `joint_column_permutation_coupling_error`: maximum `0.000e+00`, p99 `0.000e+00`, failures `0`/`10000`, tolerance `1.0e-08`.
- `joint_column_permutation_logit_error`: maximum `0.000e+00`, p99 `0.000e+00`, failures `0`/`10000`, tolerance `1.0e-08`.
- `joint_column_permutation_probability_error`: maximum `0.000e+00`, p99 `0.000e+00`, failures `0`/`10000`, tolerance `1.0e-08`.

Rows denote unordered incident hyperedges, so row-order invariance is mandatory.
Columns denote neighbor nodes, so joint column relabeling is mandatory. Root
swap requires cost/coupling transpose equivariance and final-score invariance.

## Limitations, robustness, and interpretation

- The corpus is synthetic and targets the local incidence-star criterion from
  the cited formulation; it does not establish performance on arbitrary
  empirical hypergraphs or every hierarchy level in Eidi et al.
- WL digests are screening identifiers, not proofs of graph isomorphism.
  Candidates with ambiguous positive/negative digests were rejected, and exact
  labels use multiset equality plus Hungarian cost rather than WL hashes.
- Five seeds quantify optimization variability on fixed data, not uncertainty
  over an external data-generating population.
- Strong OOD performance, if present, supports this locked generator family;
  it is not a causal claim or a universal theorem about learned matchers.

## Reproduction and evidence

Branch: `codex-phase3-final`. Baseline commit: `119343d0ad758f8c4a78d3831e8727d37255a3be`. Python `3.11.15`
via the documented project environment; PyTorch `2.12.1+cpu` on CPU.
The preserved pre-takeover archive is
`results/phase3_archive/20260727_102315`.

Run the logged clean pipeline from the repository root using the interpreter
recorded in `results/phase3/run_manifest.json`. Evidence is in
`per_seed_metrics.csv`, `aggregate_metrics.csv`, the gate/audit JSON files,
`checkpoints/`, `predictions/`, both clean-executed notebooks, and the final
artifact manifest.

## Recommended next steps

Use the locked artifacts as the reproducible Phase 3 reference. Any follow-up
should add genuinely new generator families or empirical incidence stars while
preserving validation-only selection and rerunning the same structural audit.

## Further questions

The most useful next scientific question is whether the ranking and OOD-size
behavior persist under independently designed generators that preserve the
same formal class definitions but alter degree, sparsity, and automorphism
distributions.
