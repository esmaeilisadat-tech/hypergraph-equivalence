# Final v2 results

Saved results, mean ± sample standard deviation across five seeds (42, 100, 2026, 777, 999).

| Model | Role | Split | n per seed | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC | Loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `LearnedSinkhornMatcher` | `primary_learned` | `iid_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000008 ± 0.000005 |
| `LearnedSinkhornMatcher` | `primary_learned` | `ood_template_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000008 ± 0.000004 |
| `LearnedSinkhornMatcher` | `primary_learned` | `ood_size_test` | 400 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.004026 ± 0.003397 |
| `MLP_Baseline` | `baseline` | `iid_test` | 800 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.494946 ± 0.005656 | 0.500000 ± 0.000000 | 0.693544 ± 0.000694 |
| `MLP_Baseline` | `baseline` | `ood_template_test` | 800 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.492580 ± 0.007803 | 0.500000 ± 0.000000 | 0.693517 ± 0.000645 |
| `MLP_Baseline` | `baseline` | `ood_size_test` | 400 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.394586 ± 0.041705 | 0.500000 ± 0.000000 | 0.693377 ± 0.000141 |
| `NestedDeepSets` | `baseline` | `iid_test` | 800 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.486710 ± 0.010991 | 0.499976 ± 0.000033 | 0.693148 ± 0.000001 |
| `NestedDeepSets` | `baseline` | `ood_template_test` | 800 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.491429 ± 0.005872 | 0.500016 ± 0.000007 | 0.693148 ± 0.000001 |
| `NestedDeepSets` | `baseline` | `ood_size_test` | 400 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.470502 ± 0.029633 | 0.500005 ± 0.000048 | 0.693147 ± 0.000000 |
| `BipartiteEncoder` | `baseline` | `iid_test` | 800 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.491278 ± 0.011727 | 0.500003 ± 0.000005 | 0.693150 ± 0.000004 |
| `BipartiteEncoder` | `baseline` | `ood_template_test` | 800 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.488569 ± 0.015260 | 0.499998 ± 0.000007 | 0.693150 ± 0.000003 |
| `BipartiteEncoder` | `baseline` | `ood_size_test` | 400 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.442169 ± 0.047293 | 0.499995 ± 0.000026 | 0.693149 ± 0.000002 |
| `DeterministicDifferentiableMatching` | `algorithmic_baseline` | `iid_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.064438 ± 0.000000 |
| `DeterministicDifferentiableMatching` | `algorithmic_baseline` | `ood_template_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.064218 ± 0.000000 |
| `DeterministicDifferentiableMatching` | `algorithmic_baseline` | `ood_size_test` | 400 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.094802 ± 0.000000 |
| `ExactHungarianOracle` | `oracle` | `iid_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| `ExactHungarianOracle` | `oracle` | `ood_template_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| `ExactHungarianOracle` | `oracle` | `ood_size_test` | 400 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| `FixedL1ResidualSinkhorn` | `algorithmic_hybrid` | `iid_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.064437 ± 0.000001 |
| `FixedL1ResidualSinkhorn` | `algorithmic_hybrid` | `ood_template_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.064218 ± 0.000001 |
| `FixedL1ResidualSinkhorn` | `algorithmic_hybrid` | `ood_size_test` | 400 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.094801 ± 0.000024 |
| `FixedL1StraightThroughHungarian` | `algorithmic_baseline` | `iid_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.063923 ± 0.000000 |
| `FixedL1StraightThroughHungarian` | `algorithmic_baseline` | `ood_template_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.063715 ± 0.000000 |
| `FixedL1StraightThroughHungarian` | `algorithmic_baseline` | `ood_size_test` | 400 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.063782 ± 0.000000 |
| `LearnedNoSinkhorn` | `ablation` | `iid_test` | 800 | 0.998750 ± 0.000000 | 0.998750 ± 0.000000 | 0.998750 ± 0.000000 | 1.000000 ± 0.000000 | 0.003221 ± 0.000647 |
| `LearnedNoSinkhorn` | `ablation` | `ood_template_test` | 800 | 0.998750 ± 0.000000 | 0.998750 ± 0.000000 | 0.998750 ± 0.000000 | 1.000000 ± 0.000000 | 0.002739 ± 0.000736 |
| `LearnedNoSinkhorn` | `ablation` | `ood_size_test` | 400 | 0.568500 ± 0.009937 | 0.568500 ± 0.009937 | 0.469574 ± 0.017862 | 0.998720 ± 0.000906 | 3.070588 ± 0.338244 |
| `LearnedNoColumnContext` | `ablation` | `iid_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000009 ± 0.000003 |
| `LearnedNoColumnContext` | `ablation` | `ood_template_test` | 800 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000010 ± 0.000003 |
| `LearnedNoColumnContext` | `ablation` | `ood_size_test` | 400 | 0.998500 ± 0.003354 | 0.998500 ± 0.003354 | 0.998500 ± 0.003354 | 1.000000 ± 0.000000 | 0.008121 ± 0.012379 |

Source: [aggregate_metrics.csv](../../results/phase3/aggregate_metrics.csv).

The primary is specification-designated, not chosen from the test ranking. It is the best learned model by mean macro F1 over these three test splits. Four algorithmic/oracle comparators tie its perfect classification scores. The generated presentation check recomputes accuracy, balanced accuracy, macro F1 and ROC-AUC from all saved predictions, recomputes every aggregate column from per-seed metrics, and verifies checkpoint and prediction hashes. Loss is aggregated from saved per-seed values; it is not re-evaluated from checkpoint inference in this documentation pass.
