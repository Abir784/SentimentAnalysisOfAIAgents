# Ensemble spec: VADER + SentiWordNet (unanimity with neutral fallback)

Source of truth: `scripts/run_moltbook_rule_based.py`.

## Inputs (per comment, raw `text`)

- `vader_compound`: `nltk.sentiment.vader.SentimentIntensityAnalyzer.polarity_scores(text)["compound"]`, no scaling.
- `swn_score`: mean over tokens of first-synset `pos_score - neg_score`
  (`_sentiwordnet_score`, `scripts/run_moltbook_rule_based.py:67-90`):
  lowercase, `TOKEN_RE` tokenize, POS-tag, lemmatize, `swn.senti_synsets(lemma, wn_pos)[0]`.
  No scaling, no weights; empty/no-synset text scores `0.0`.

## Thresholding (`_label_from_score`, lines 59-64)

```
positive if value >= +threshold
negative if value <= -threshold
neutral  otherwise
```

Defaults: `--vader-threshold 0.05`, `--swn-threshold 0.02` (lines 184-195).

## Combination rule (lines 248-255) — unanimity, NOT voting/averaging

```
if vader_label == swn_label: return that label
else: return "neutral"
```

`method_agreement = (vader_label == swn_label)`. Earlier docs saying
"voting"/"majority voting" (`generate_rq4_robustness_report.py:85,193,225`)
are loose wording for this rule; fixed here.

## Dev-split tuning protocol (gold never used as test for tuning)

1. Take adjudicated gold when available (currently: provisional single-rater
   `final_label`, n=250 — see `data/gold/provisional_benchmark_250.json`).
2. Stratified 50/50 split by gold label, seed 42 → dev / test.
3. Grid-search `(vader_t, swn_t)` on **dev only**, objective macro-F1.
4. Report default thresholds vs tuned thresholds on held-out **test**.

### Provisional run (single-rater reference, 2026-10-16)

Grid: `vader_t in {0.0, 0.05, 0.1, 0.2, 0.3}`, `swn_t in {0.0, 0.02, 0.05, 0.1}`.
Script: `scripts/provisional_benchmark_and_stats.py::ensemble_tuning`.

- dev n=124, test n=126.
- Best on dev: `vader_t=0.0, swn_t=0.0`, dev macro-F1 0.390.
- On held-out test: default (0.05/0.02) macro-F1 **0.422**, acc 0.429;
  tuned (0.0/0.0) macro-F1 **0.459**, acc 0.484.
- Direction: lower thresholds help the ensemble slightly on this sample
  (less neutral fallback), but absolute scores stay below both RoBERTa models
  (see benchmark table). Do **not** adopt tuned thresholds as final until the
  adjudicated gold exists; re-run this protocol on adjudicated labels.

## Provisional benchmark (single-rater gold, n=250)

| scorer | accuracy | macro-F1 |
|---|---|---|
| twitter-roberta | 0.448 | **0.451** |
| bertweet | 0.452 | 0.440 |
| xlm-twitter | 0.392 | 0.392 |
| SentiWordNet | 0.400 | 0.377 |
| VADER | 0.480 | 0.372 |
| ensemble (unanimity) | 0.388 | 0.368 |

Full counts in `data/gold/provisional_benchmark_250.json`.
Provisional ranking by macro-F1: twitter-roberta > bertweet > xlm-twitter >
SWN > VADER > ensemble. The ensemble must **not** be the default "correct"
scorer; choose the main scorer from adjudicated-gold results, not in advance.
