# Hypergraph Equivalence with Learned Sinkhorn Matching

**Mahdieh Sadat Esmaeili**

A research project on learning local structural equivalence in hypergraphs using neural matching costs and balanced Sinkhorn transport.

[Research paper](docs/paper/Hypergraph_Sinkhorn_Paper_LaTeX_V3.pdf)

## Project files

| Directory | Contents |
|---|---|
| `src/` | Models, data generation, training and evaluation |
| `configs/` | Model configurations |
| `data/splits/` | Locked final experimental splits |
| `notebooks/` | Final research notebooks in English and Persian |
| `results/phase3/` | Checkpoints, predictions, metrics and scientific evaluation evidence |
| `tests/` | Repository tests |

The primary model is `LearnedSinkhornMatcher`. The final synthetic benchmark evaluates five seeds: `42`, `100`, `2026`, `777`, `999`, with separate IID, OOD-template and OOD-size tests. Results apply to this local synthetic incidence-star task, not unrestricted hypergraph isomorphism.

## Run

Use Python 3.11 and the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.final_checks
```

The final notebooks are [Phase3_Research.ipynb](notebooks/Phase3_Research.ipynb) and [Phase3_Explained_FA.ipynb](notebooks/Phase3_Explained_FA.ipynb). Detailed results are in [aggregate_metrics.csv](results/phase3/aggregate_metrics.csv) and [final_validation_report.md](results/phase3/final_validation_report.md).

Historical deterministic and development runs are stored separately from the active `results/phase3/` study. Before rerunning commands that write experimental outputs, back up the current evidence.
