# Reproduction guide

Pinned environment: `requirements.txt` (direct) / `requirements_frozen_direct.txt`;
transformers work: `requirements_transformers.txt`. Use a fresh venv —
do not commit it (`.venv/` is git-ignored and untracked).

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
# transformers (CPU default; GPU wheels per PyTorch site):
python -m pip install -r requirements_transformers.txt
```

## End-to-end (rule-based + RQ)

```powershell
python scripts/run_nlp_pipeline.py run --all
python scripts/run_moltbook_rule_based.py
python scripts/run_moltbook_interaction_network.py --edge-mode auto
python scripts/run_moltbook_rq1_graph_metrics.py
python scripts/run_moltbook_rq2_stats.py --bootstrap 4000 --seed 42
python scripts/run_moltbook_rq3_sentiment_dynamics.py
python scripts/run_moltbook_rq4_robustness.py
```

## Gold standard (human annotation required)

```powershell
python scripts/build_moltbook_goldset_sample.py --target-size 400 --seed 42
# annotate data/gold/moltbook_goldset_sample_<RUN_ID>.csv per data/gold/ANNOTATION_GUIDE.md
# rater-2 blind overlap batch (49 rows): data/gold/rater2_overlap_batch_250.csv
python scripts/evaluate_moltbook_goldset.py --input data/gold/moltbook_goldset_sample_<RUN_ID>.csv
# provisional (single-rater) six-scorer benchmark + stats recomputation:
python3 scripts/provisional_benchmark_and_stats.py
# transformer labels for the 250 (reuse-first, exact benchmark settings):
python3 scripts/get_transformer_labels_250.py
```

## Key docs

- Preregistration/design matrix:
  `moltbook_research_design_matrix_and_preregistration.md` (cited from README).
- Ensemble recipe + tuning: `docs/ensemble_spec.md`.
- Stats errata (df, effect sizes, lifts, trend, uniformity dropped):
  `docs/stats_errata_and_fixes.md`.
- Annotation protocol: `data/gold/ANNOTATION_GUIDE.md`.
- License: `LICENSE` (MIT).
