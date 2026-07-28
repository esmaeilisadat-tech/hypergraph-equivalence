# Phase 3 Learned Hypergraph Equivalence — Final Validation Report

## Technical summary

**Overall validated status: PASS.** The specification-designated primary
is `LearnedSinkhornMatcher` (`primary_learned`). Its final-v2 result is IID accuracy 1.000000 ± 0.000000, macro F1 1.000000 ± 0.000000, ROC-AUC 1.000000 ± 0.000000; OOD-size accuracy 1.000000 ± 0.000000, macro F1 1.000000 ± 0.000000, ROC-AUC 1.000000 ± 0.000000; OOD-template accuracy 1.000000 ± 0.000000, macro F1 1.000000 ± 0.000000, ROC-AUC 1.000000 ± 0.000000.
The numerically best learned model(s), ranked by mean macro F1 across IID,
OOD-template, and OOD-size, are `LearnedSinkhornMatcher` at
`1.000000`. Across learned and algorithmic/oracle rows, the
best are `DeterministicDifferentiableMatching`, `ExactHungarianOracle`, `FixedL1ResidualSinkhorn`, `FixedL1StraightThroughHungarian`, `LearnedSinkhornMatcher` (exact tie) at `1.000000`.

The learned primary is not the old fixed-L1 solver. It starts from random
weights, is at chance before training, has independently seeded initial/final
states and training orders, and improves mean training accuracy by
`0.500000`. Fixed-cost and exact-Hungarian results are preserved
as algorithmic evidence and are never attributed to learned discovery.

## Key findings

- All five `LearnedSinkhornMatcher` seeds were optimized with validation-only checkpoint
  selection and evaluated on the same locked final-v2 splits.
- Primary zero-shot accuracy spans
  `0.500000`–`0.500000`
  across 25 seed/split rows; trained primary mean train accuracy is
  `1.000000`.
- Final-v2 contains 800/800/800/800/400 balanced samples in
  train/validation/IID/OOD-template/OOD-size, has zero prohibited overlap, and
  has no exact-oracle label disagreement.
- Every required scientific, structural, numerical, and reproducibility gate
  used to generate this report has status `PASS` as itemized below.

## Learned models and ablations

| Model | Role | Split | Loss | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC | Seeds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LearnedNoColumnContext | ablation | IID | 0.000009 ± 0.000003 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| LearnedNoColumnContext | ablation | OOD-size | 0.008121 ± 0.012379 | 0.998500 ± 0.003354 | 0.998500 ± 0.003354 | 0.998500 ± 0.003354 | 1.000000 ± 0.000000 | 5 |
| LearnedNoColumnContext | ablation | OOD-template | 0.000010 ± 0.000003 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| LearnedNoSinkhorn | ablation | IID | 0.003221 ± 0.000647 | 0.998750 ± 0.000000 | 0.998750 ± 0.000000 | 0.998750 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| LearnedNoSinkhorn | ablation | OOD-size | 3.070588 ± 0.338244 | 0.568500 ± 0.009937 | 0.568500 ± 0.009937 | 0.469574 ± 0.017862 | 0.998720 ± 0.000906 | 5 |
| LearnedNoSinkhorn | ablation | OOD-template | 0.002739 ± 0.000736 | 0.998750 ± 0.000000 | 0.998750 ± 0.000000 | 0.998750 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| BipartiteEncoder | baseline | IID | 0.693150 ± 0.000004 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.491278 ± 0.011727 | 0.500003 ± 0.000005 | 5 |
| BipartiteEncoder | baseline | OOD-size | 0.693149 ± 0.000002 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.442169 ± 0.047293 | 0.499995 ± 0.000026 | 5 |
| BipartiteEncoder | baseline | OOD-template | 0.693150 ± 0.000003 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.488569 ± 0.015260 | 0.499998 ± 0.000007 | 5 |
| MLP_Baseline | baseline | IID | 0.693544 ± 0.000694 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.494946 ± 0.005656 | 0.500000 ± 0.000000 | 5 |
| MLP_Baseline | baseline | OOD-size | 0.693377 ± 0.000141 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.394586 ± 0.041705 | 0.500000 ± 0.000000 | 5 |
| MLP_Baseline | baseline | OOD-template | 0.693517 ± 0.000645 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.492580 ± 0.007803 | 0.500000 ± 0.000000 | 5 |
| NestedDeepSets | baseline | IID | 0.693148 ± 0.000001 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.486710 ± 0.010991 | 0.499976 ± 0.000033 | 5 |
| NestedDeepSets | baseline | OOD-size | 0.693147 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.470502 ± 0.029633 | 0.500005 ± 0.000048 | 5 |
| NestedDeepSets | baseline | OOD-template | 0.693148 ± 0.000001 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.491429 ± 0.005872 | 0.500016 ± 0.000007 | 5 |
| LearnedSinkhornMatcher | primary_learned | IID | 0.000008 ± 0.000005 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| LearnedSinkhornMatcher | primary_learned | OOD-size | 0.004026 ± 0.003397 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| LearnedSinkhornMatcher | primary_learned | OOD-template | 0.000008 ± 0.000004 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |

These rows are the only rows eligible for learned-performance claims.
`MLP_Baseline`, `NestedDeepSets`, and `BipartiteEncoder` are learned baselines;
`LearnedNoSinkhorn` and `LearnedNoColumnContext` are ablations.

## Algorithmic, hybrid, and oracle controls

| Model | Role | Split | Loss | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC | Seeds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DeterministicDifferentiableMatching | algorithmic_baseline | IID | 0.064438 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| DeterministicDifferentiableMatching | algorithmic_baseline | OOD-size | 0.094802 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| DeterministicDifferentiableMatching | algorithmic_baseline | OOD-template | 0.064218 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| FixedL1StraightThroughHungarian | algorithmic_baseline | IID | 0.063923 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| FixedL1StraightThroughHungarian | algorithmic_baseline | OOD-size | 0.063782 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| FixedL1StraightThroughHungarian | algorithmic_baseline | OOD-template | 0.063715 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| FixedL1ResidualSinkhorn | algorithmic_hybrid | IID | 0.064437 ± 0.000001 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| FixedL1ResidualSinkhorn | algorithmic_hybrid | OOD-size | 0.094801 ± 0.000024 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| FixedL1ResidualSinkhorn | algorithmic_hybrid | OOD-template | 0.064218 ± 0.000001 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| ExactHungarianOracle | oracle | IID | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| ExactHungarianOracle | oracle | OOD-size | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |
| ExactHungarianOracle | oracle | OOD-template | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 5 |

`DeterministicDifferentiableMatching`,
`FixedL1StraightThroughHungarian`, and `ExactHungarianOracle` are not trained.
`FixedL1ResidualSinkhorn` contains a parameterized residual component but is
non-optimized in the locked study and receives the fixed-L1 separator. It is
therefore an algorithmic hybrid, not evidence that a matching cost was
discovered from scratch.

<!-- HEADLINE_METRICS_JSON_START -->
[{"mean": 8.076873781895485e-06, "metric": "loss", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "iid_test", "std": 4.556830588707213e-06}, {"mean": 1.0, "metric": "accuracy", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "iid_test", "std": 1.1102230246251563e-16}, {"mean": 1.9695914760253784e-10, "metric": "brier_score", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "iid_test", "std": 1.7897998004654292e-10}, {"mean": 8.280781531712364e-06, "metric": "loss", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_template_test", "std": 4.206681827433285e-06}, {"mean": 1.0, "metric": "accuracy", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_template_test", "std": 1.2412670766236366e-16}, {"mean": 2.1499056424012512e-09, "metric": "brier_score", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_template_test", "std": 1.6020977790133602e-09}, {"mean": 0.0040259650996774, "metric": "loss", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_size_test", "std": 0.0033972603590646}, {"mean": 1.0, "metric": "accuracy", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.0002616827832753, "metric": "brier_score", "model": "LearnedSinkhornMatcher", "role": "primary_learned", "seed_count": 5, "split": "ood_size_test", "std": 0.0002885851780556}, {"mean": 0.6935439279583846, "metric": "loss", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0006943710262198}, {"mean": 0.5, "metric": "accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.4949455968294082, "metric": "macro_f1", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0056563410511592}, {"mean": 0.5, "metric": "roc_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 3.925231146709438e-17}, {"mean": 0.5, "metric": "pr_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.2501972138881683, "metric": "brier_score", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0003446289162863}, {"mean": 0.6935173839413479, "metric": "loss", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0006446765071844}, {"mean": 0.5, "metric": "accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.4925797229865352, "metric": "macro_f1", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0078030203205585}, {"mean": 0.5, "metric": "roc_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 6.206335383118183e-17}, {"mean": 0.5, "metric": "pr_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.2501842021942139, "metric": "brier_score", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0003203456065398}, {"mean": 0.6933765700454811, "metric": "loss", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0001412712169069}, {"mean": 0.5, "metric": "accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.3945864522762451, "metric": "macro_f1", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.04170464008104}, {"mean": 0.5, "metric": "roc_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.5, "metric": "pr_auc", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.2501145124435425, "metric": "brier_score", "model": "MLP_Baseline", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 7.047132230661389e-05}, {"mean": 0.6931477467388065, "metric": "loss", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 6.090481613292321e-07}, {"mean": 0.5, "metric": "accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.486709983788732, "metric": "macro_f1", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0109908319971261}, {"mean": 0.49997625, "metric": "roc_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 3.252102685184793e-05}, {"mean": 0.5000123211680579, "metric": "pr_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 3.3351206428717734e-05}, {"mean": 0.2500002861022949, "metric": "brier_score", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 3.036328703668118e-07}, {"mean": 0.6931476942731921, "metric": "loss", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 5.523624117517546e-07}, {"mean": 0.5, "metric": "accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.4914285257891604, "metric": "macro_f1", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0058723591503106}, {"mean": 0.50001625, "metric": "roc_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 7.126096406867002e-06}, {"mean": 0.5000492204126517, "metric": "pr_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 4.981589527033842e-05}, {"mean": 0.250000250339508, "metric": "brier_score", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 2.7444039424000976e-07}, {"mean": 0.6931473966918162, "metric": "loss", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 2.2658870215086747e-07}, {"mean": 0.5, "metric": "accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.4705015610627049, "metric": "macro_f1", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0296328689460406}, {"mean": 0.500005, "metric": "roc_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 4.808846015418605e-05}, {"mean": 0.500046518475078, "metric": "pr_auc", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 5.341567776115719e-05}, {"mean": 0.2500001013278961, "metric": "brier_score", "model": "NestedDeepSets", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 1.1657242307092229e-07}, {"mean": 0.6931503141623672, "metric": "loss", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 3.69203078497098e-06}, {"mean": 0.5, "metric": "accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.4912780439128375, "metric": "macro_f1", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 0.011727411923561}, {"mean": 0.500003125, "metric": "roc_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 4.941058843988411e-06}, {"mean": 0.5000128396142409, "metric": "pr_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 1.4877301607615634e-05}, {"mean": 0.2500015676021576, "metric": "brier_score", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "iid_test", "std": 1.855012035384316e-06}, {"mean": 0.693149990289573, "metric": "loss", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 3.338877228045585e-06}, {"mean": 0.5, "metric": "accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.4885690315236747, "metric": "macro_f1", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0152597942046845}, {"mean": 0.499998125, "metric": "roc_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 7.194290270776208e-06}, {"mean": 0.5000109099823684, "metric": "pr_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 7.724235127806281e-06}, {"mean": 0.2500014007091522, "metric": "brier_score", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_template_test", "std": 1.6747738502244797e-06}, {"mean": 0.6931488392500447, "metric": "loss", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 1.6534969114629565e-06}, {"mean": 0.5, "metric": "accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.5, "metric": "balanced_accuracy", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.4421692894106863, "metric": "macro_f1", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0472933747905664}, {"mean": 0.499995, "metric": "roc_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 2.5920551691638055e-05}, {"mean": 0.5000682197911627, "metric": "pr_auc", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 6.958533995322386e-05}, {"mean": 0.2500008404254913, "metric": "brier_score", "model": "BipartiteEncoder", "role": "baseline", "seed_count": 5, "split": "ood_size_test", "std": 8.305709955120667e-07}, {"mean": 0.0644375061714888, "metric": "loss", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.0071940911002457, "metric": "brier_score", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.0642176339513121, "metric": "loss", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.0071986722759902, "metric": "brier_score", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.094801778756211, "metric": "loss", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.0161385852843523, "metric": "brier_score", "model": "DeterministicDifferentiableMatching", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0310768360851038e-09, "metric": "loss", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0620885327846651e-18, "metric": "brier_score", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0310768360851038e-09, "metric": "loss", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0620885327846651e-18, "metric": "brier_score", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0310768360851036e-09, "metric": "loss", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0620885327846651e-18, "metric": "brier_score", "model": "ExactHungarianOracle", "role": "oracle", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.0644374960993423, "metric": "loss", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "iid_test", "std": 5.106830637753491e-07}, {"mean": 1.0, "metric": "accuracy", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.0071940885856747, "metric": "brier_score", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "iid_test", "std": 1.1225750265760283e-07}, {"mean": 0.0642176223246586, "metric": "loss", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_template_test", "std": 5.483599249201682e-07}, {"mean": 1.0, "metric": "accuracy", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.0071986699476838, "metric": "brier_score", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_template_test", "std": 1.195530376847896e-07}, {"mean": 0.0948005229075105, "metric": "loss", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_size_test", "std": 2.4423195062494165e-05}, {"mean": 1.0, "metric": "accuracy", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.016138120368123, "metric": "brier_score", "model": "FixedL1ResidualSinkhorn", "role": "algorithmic_hybrid", "seed_count": 5, "split": "ood_size_test", "std": 8.658216176462142e-06}, {"mean": 0.0639230549656122, "metric": "loss", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.0071053123101592, "metric": "brier_score", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.063714722918534, "metric": "loss", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.0071049975231289, "metric": "brier_score", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.06378226400948, "metric": "loss", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "accuracy", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.0071053802967071, "metric": "brier_score", "model": "FixedL1StraightThroughHungarian", "role": "algorithmic_baseline", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.0032212610471091, "metric": "loss", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0006472799592996}, {"mean": 0.99875, "metric": "accuracy", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.99875, "metric": "balanced_accuracy", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.998749998046872, "metric": "macro_f1", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 0.0010116024524904, "metric": "brier_score", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 8.80761241900704e-05}, {"mean": 0.0027391947453447, "metric": "loss", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0007360692378387}, {"mean": 0.99875, "metric": "accuracy", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.99875, "metric": "balanced_accuracy", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.998749998046872, "metric": "macro_f1", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 0.0009213633020408, "metric": "brier_score", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0002408010881245}, {"mean": 3.070587608104694, "metric": "loss", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.3382441698001769}, {"mean": 0.5685, "metric": "accuracy", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0099373034571758}, {"mean": 0.5685, "metric": "balanced_accuracy", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0099373034571758}, {"mean": 0.4695740223473729, "metric": "macro_f1", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0178623166787845}, {"mean": 0.99872, "metric": "roc_auc", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.000906469800931}, {"mean": 0.9986061894857612, "metric": "pr_auc", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0010217986604418}, {"mean": 0.4107056140899658, "metric": "brier_score", "model": "LearnedNoSinkhorn", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0119191703818315}, {"mean": 9.272818181283792e-06, "metric": "loss", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 2.7519922866045463e-06}, {"mean": 1.0, "metric": "accuracy", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 9.614813431917819e-17}, {"mean": 2.39169328786204e-10, "metric": "brier_score", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "iid_test", "std": 1.4338438934744646e-10}, {"mean": 9.614834403389691e-06, "metric": "loss", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 2.865561752052891e-06}, {"mean": 1.0, "metric": "accuracy", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "balanced_accuracy", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "macro_f1", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 7.850462293418876e-17}, {"mean": 2.2798995025219428e-09, "metric": "brier_score", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_template_test", "std": 1.0820916747957636e-09}, {"mean": 0.0081212404089965, "metric": "loss", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0123787239371165}, {"mean": 0.9985, "metric": "accuracy", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0033541019662496}, {"mean": 0.9985, "metric": "balanced_accuracy", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0033541019662497}, {"mean": 0.9984999156202538, "metric": "macro_f1", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0033542906450984}, {"mean": 1.0, "metric": "roc_auc", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 1.0, "metric": "pr_auc", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0}, {"mean": 0.0016404011649342, "metric": "brier_score", "model": "LearnedNoColumnContext", "role": "ablation", "seed_count": 5, "split": "ood_size_test", "std": 0.0033785048825581}]
<!-- HEADLINE_METRICS_JSON_END -->

## Scope, data definitions, and benchmark difficulty

The binary task distinguishes EE-equivalent incidence stars that are not
strongly structurally equivalent (class 0) from stars whose incident-neighbor
multisets are exactly equal (class 1). Matched positive/negative pairs share
dimensions, margins, density, and total incidence. The nuisance-scalar audit
has maximum univariate direction-adjusted AUC
`0.500000000000`.
Negative exact Hungarian costs range from
`4.0`
to
`20.0`;
positive costs are exactly zero by definition.

Split integrity is `PASS` with
`0` problems and zero prohibited sample, matched
pair, base, or invariant-structure overlap. IDs, dimensions, labels, masks,
padding, deterministic regeneration, and file hashes were audited.

## Model-selection chronology and test-data remedy

Candidate v1 failed and its generic development runner exposed v1 test
metrics. Those test partitions are preserved but explicitly retired from final
claims. Candidates v2–v5 were judged on train/validation evidence; v5 was the
first to meet unchanged Gate C thresholds and was selected with validation
accuracy/macro F1
`1.000000/`
`1.000000`.

After selection, the model source, training source, configuration, and
selection specification were hashed, and entirely new final-v2 IID,
OOD-template, and OOD-size samples were generated at unseen offsets. The
scientifically accurate lock claim is:
`locked_before_first_final_test_evaluation =
true`;
`locked_before_first_optimization =
false`.
Final-v2 test data did not guide model or hyperparameter selection. This
disclosure supersedes the broader, inaccurate policy sentence retained in the
original specification.

## Learned-primary architecture and training

`LearnedSinkhornMatcher` uses a randomly initialized symmetric column-pair MLP, learned
row-pair scores converted to non-negative costs, balanced log-domain Sinkhorn
transport, transport summaries, and a learned classifier. Its forward
signature receives only `B_x` and `B_y`. It never computes or receives fixed
L1/Hamming cost, exact Hungarian output, labels, IDs, hashes, or split names.

Cost pretraining uses only train rows paired with themselves versus a single
random incidence-bit flip and selects on analogous synthetic validation-row
BCE. Main checkpoint selection is validation macro F1 followed by validation
loss. Test metrics never select epochs or temperatures.

## Zero-shot, label-signal, and label-access controls

The zero-shot primary must remain within
`0.35`–`0.65`
accuracy. The valid supervised-label-signal controls disable only the
label-free structural pretraining, corrupt train and validation labels, retain
the locked architecture/optimizer/seeds/splits, and evaluate untouched true
test labels:

- random labels, IID: accuracy 0.489750 ± 0.090357; macro F1 0.486514; seeds 5.
- random labels, OOD-size: accuracy 0.500000 ± 0.000000; macro F1 0.333333; seeds 5.
- random labels, OOD-template: accuracy 0.494250 ± 0.116641; macro F1 0.490928; seeds 5.
- shuffled labels, IID: accuracy 0.494250 ± 0.020128; macro F1 0.489143; seeds 5.
- shuffled labels, OOD-size: accuracy 0.500000 ± 0.000000; macro F1 0.333333; seeds 5.
- shuffled labels, OOD-template: accuracy 0.503250 ± 0.031962; macro F1 0.494272; seeds 5.

The initially completed full-pipeline control is preserved, not discarded.
It failed the 0.65 ceiling because label-free structural pretraining itself
supplies genuine task signal, so it cannot isolate supervised label signal:

- random labels, IID: mean accuracy 0.665000.
- random labels, OOD-size: mean accuracy 0.522000.
- random labels, OOD-template: mean accuracy 0.691500.
- shuffled labels, IID: mean accuracy 0.369000.
- shuffled labels, OOD-size: mean accuracy 0.524500.
- shuffled labels, OOD-template: mean accuracy 0.367500.

The label-access positive control achieves the expected accuracy
`1.0` on every split.
Source inspection finds
`0` prohibited forward
tokens, and the real model receives only `B_x`, `B_y`.

## Gate A: oracle and normalized coupling

Gate A is `PASS` on `3600` samples
with oracle accuracy `1.000000`. Coupling mass is
normalized to one, so a hard matched entry for `m` equally weighted rows is
approximately `1/m`, not one. At 500 float64 iterations, maximum total-mass,
row-marginal, and column-marginal errors are
`2.220e-16`,
`7.354e-14`, and
`7.297e-14` at tolerance
`1.0e-08`.

## Gate B: unchanged five-seed tiny-batch rule

- Seed 42: **PASS**; accuracy 1.000000; macro F1 1.000000; loss 0.00050928; failed IDs none.
- Seed 100: **PASS**; accuracy 1.000000; macro F1 1.000000; loss 0.00041490; failed IDs none.
- Seed 2026: **PASS**; accuracy 1.000000; macro F1 1.000000; loss 0.00039782; failed IDs none.
- Seed 777: **PASS**; accuracy 1.000000; macro F1 1.000000; loss 0.00047787; failed IDs none.
- Seed 999: **PASS**; accuracy 1.000000; macro F1 1.000000; loss 0.00039347; failed IDs none.

The acceptance rule remains accuracy ≥ 0.99, macro F1 ≥ 0.99, and each class
F1 ≥ 0.98. No threshold or sample was removed after observing results.

## Gate C: validation-only checkpoint selection

- Seed 42: **PASS**; initial/train accuracy 0.500000/1.000000; train macro F1 1.000000; selected epoch 114; temperature 0.01997823; first-order hash `fd333efc389dc499daf0400fa3e38b0f4bdf3e0fb3efdacd28398918ee3ada8e`.
- Seed 100: **PASS**; initial/train accuracy 0.500000/1.000000; train macro F1 1.000000; selected epoch 96; temperature 0.02309215; first-order hash `890ba39d5075d67cb8afb6729311fd2e89990927885362424cd3f6687c698b70`.
- Seed 2026: **PASS**; initial/train accuracy 0.500000/1.000000; train macro F1 1.000000; selected epoch 135; temperature 0.01687195; first-order hash `af808f43b982abbd4b0d7320c99787ce38efba484628a98977c152b333402910`.
- Seed 777: **PASS**; initial/train accuracy 0.500000/1.000000; train macro F1 1.000000; selected epoch 160; temperature 0.01379730; first-order hash `01e389b32d5f08f4d02b590404b0a3cb8fe7e9d62c579367142529947679bd58`.
- Seed 999: **PASS**; initial/train accuracy 0.500000/1.000000; train macro F1 1.000000; selected epoch 135; temperature 0.01687195; first-order hash `fc46667a36900548b15538a461a34c473a2a485c0bfa937ecc44405951261c81`.

The unchanged train thresholds are accuracy, balanced accuracy, and macro F1
at least 0.95, with each class recall at least 0.90.

## Complete float64 structural audit

The numerical audit is `PASS` over
`10000` seed–sample cases with
`0` failed cases, compact unpadded matrices,
explicit uniform marginals, deterministic permutations, float64, and
`500` Sinkhorn iterations.

- `cost_finite_violation`: maximum `0.000e+00`, p95 `0.000e+00`, p99 `0.000e+00`, failures `0/10000`, tolerance `1.0e-08`.
- `coupling_finite_violation`: maximum `0.000e+00`, p95 `0.000e+00`, p99 `0.000e+00`, failures `0/10000`, tolerance `1.0e-08`.
- `coupling_nonnegativity_violation`: maximum `0.000e+00`, p95 `0.000e+00`, p99 `0.000e+00`, failures `0/10000`, tolerance `1.0e-08`.
- `cost_transpose_error`: maximum `1.137e-13`, p95 `0.000e+00`, p99 `4.441e-16`, failures `0/10000`, tolerance `1.0e-08`.
- `coupling_transpose_error`: maximum `1.469e-13`, p95 `4.205e-14`, p99 `6.931e-14`, failures `0/10000`, tolerance `1.0e-08`.
- `assignment_transpose_error`: maximum `1.469e-13`, p95 `4.205e-14`, p99 `6.931e-14`, failures `0/10000`, tolerance `1.0e-08`.
- `matching_cost_swap_error`: maximum `2.440e-12`, p95 `4.734e-13`, p99 `8.483e-13`, failures `0/10000`, tolerance `1.0e-08`.
- `equality_score_swap_error`: maximum `2.001e-13`, p95 `4.379e-15`, p99 `2.888e-14`, failures `0/10000`, tolerance `1.0e-08`.
- `final_logit_swap_error`: maximum `0.000e+00`, p95 `0.000e+00`, p99 `0.000e+00`, failures `0/10000`, tolerance `1.0e-08`.
- `final_probability_swap_error`: maximum `0.000e+00`, p95 `0.000e+00`, p99 `0.000e+00`, failures `0/10000`, tolerance `1.0e-08`.
- `total_coupling_mass_error`: maximum `5.551e-16`, p95 `2.220e-16`, p99 `2.220e-16`, failures `0/10000`, tolerance `1.0e-08`.
- `row_marginal_residual`: maximum `1.207e-13`, p95 `7.336e-14`, p99 `9.591e-14`, failures `0/10000`, tolerance `1.0e-08`.
- `column_marginal_residual`: maximum `1.153e-13`, p95 `7.085e-14`, p99 `9.472e-14`, failures `0/10000`, tolerance `1.0e-08`.
- `node_row_permutation_coupling_error`: maximum `4.666e-14`, p95 `4.163e-17`, p99 `8.327e-17`, failures `0/10000`, tolerance `1.0e-08`.
- `node_row_permutation_logit_error`: maximum `9.744e-10`, p95 `9.091e-12`, p99 `6.505e-11`, failures `0/10000`, tolerance `1.0e-08`.
- `node_row_permutation_probability_error`: maximum `4.885e-15`, p95 `2.220e-16`, p99 `7.082e-16`, failures `0/10000`, tolerance `1.0e-08`.
- `joint_column_permutation_cost_error`: maximum `1.137e-13`, p95 `0.000e+00`, p99 `3.331e-16`, failures `0/10000`, tolerance `1.0e-08`.
- `joint_column_permutation_coupling_error`: maximum `9.237e-14`, p95 `0.000e+00`, p99 `0.000e+00`, failures `0/10000`, tolerance `1.0e-08`.
- `joint_column_permutation_logit_error`: maximum `1.257e-09`, p95 `0.000e+00`, p99 `0.000e+00`, failures `0/10000`, tolerance `1.0e-08`.
- `joint_column_permutation_probability_error`: maximum `4.219e-15`, p95 `0.000e+00`, p99 `0.000e+00`, failures `0/10000`, tolerance `1.0e-08`.

Mandatory checks cover cost/coupling transpose equivariance, matching-cost,
equality-score, final-logit, and probability invariance, finiteness,
non-negativity, total mass, both marginals, assignment transpose, valid
incident-row permutations, and joint neighbor-column permutations.

## Validation status

- Gate A: **PASS**.
- Gate B: **PASS**.
- Gate C: **PASS**.
- split audit: **PASS**.
- benchmark audit: **PASS**.
- numerical audit: **PASS**.
- scientific-claim audit: **PASS**.
- label-access control: **PASS**.
- label-signal controls: **PASS**.
- checkpoint round trip: **PASS**.

## Limitations and interpretation

- The corpus is synthetic and tests a locked local incidence-star formulation,
  not arbitrary empirical hypergraphs or every hierarchy level in the source
  paper.
- Five seeds measure optimization variability on fixed data, not uncertainty
  over an external data-generating population.
- Exact row-multiset/Hungarian cost separates the formal labels by definition.
  The learned primary is scientifically meaningful only because that separator
  is prohibited from its inputs and implementation.
- The v1 test-exposure incident is preserved. New final-v2 tests repair final
  evaluation, but the development chronology remains an explicit limitation.
- A full-pipeline invalid-label run did not remain at chance because
  self-supervised structural pretraining can learn task-relevant matching
  without class labels. Its FAIL result is preserved; the separate
  supervised-only control isolates the narrower claim about corrupted labels.
- Strong final-v2 performance supports these locked generator families; it is
  not a universal theorem or causal claim.

## Reproduction and evidence

Branch `codex-phase3-final`; current source commit `8425f6c780ad611f2502dd7f3e57119a07f07ee0`; Python
`3.11.15`; PyTorch `2.12.1+cpu` on CPU. The
pre-correction archive is
`results/phase3_archive/20260727T160053Z_independent_audit_corrections`.
`results/phase3/fresh_environment_validation.json` records a genuinely new
temporary virtual environment: clean dependency installation, extracted-ZIP
CRC and 2,002 manifest hashes, imports, 7 repository tests, deterministic
split audit, zero-shot smoke accuracy 0.5, 800-sample checkpoint inference,
210 report metrics, and frozen-evidence checks all returned code 0; the
temporary environment was removed afterward.
Primary evidence is in `per_seed_metrics.csv`, `aggregate_metrics.csv`,
`checkpoints/`, `predictions/`, the gate/control/audit JSON files, both
clean-executed notebooks, logged commands, and the post-commit manifests.

## Recommended next work

An independent follow-up should add generator families not designed during
this project and empirical incidence stars while retaining the same
validation-only selection, zero-shot/invalid-label controls, and full
structural audit. That would test whether the learned ranking persists beyond
the locked synthetic distribution.
