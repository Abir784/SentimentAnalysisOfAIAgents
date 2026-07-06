# Moltbook Transformer Sentiment Results

Pretrained transformer sentiment scores for real Moltbook comments.

## Run context

| Field | Value |
|---|---|
| Dataset | `moltbook` |
| Input | `data/staged/moltbook_comments_all.jsonl` |
| Comments scored | 1,296 |
| Device | CUDA (NVIDIA GeForce RTX 4070 Ti SUPER) |
| Long-text strategy | `truncate` (default) |
| Social-text preprocessing | `@user` / `http` normalization applied for all three models |

Generated with:

```powershell
python scripts/run_transformer_sentiment.py moltbook --device cuda --batch-size 16 --models twitter-roberta xlm-twitter
python scripts/run_transformer_sentiment.py moltbook --device cuda --batch-size 16 --models bertweet --max-length 128
```

`twitter-roberta` and `xlm-twitter` used the script default `--max-length 512`.
`bertweet` required `--max-length 128` because BERTweet only supports ~128 tokens per sequence.

## Model results summary

| Model | Checkpoint | Rows | Max length | Mean confidence | Negative | Neutral | Positive |
|---|---|---:|---:|---:|---:|---:|---:|
| `twitter-roberta` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 1,296 | 512 | 0.692 | 276 (21.3%) | 795 (61.3%) | 225 (17.4%) |
| `xlm-twitter` | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 1,296 | 512 | 0.539 | 536 (41.4%) | 582 (44.9%) | 178 (13.7%) |
| `bertweet` | `finiteautomata/bertweet-base-sentiment-analysis` | 1,296 | 128 | 0.766 | 147 (11.3%) | 874 (67.4%) | 275 (21.2%) |

All three models use the same 3-class label space: `negative`, `neutral`, `positive`.

## Cross-model comparison

- **Dominant label:** All three models assign **neutral** to the largest share of comments.
- **Most negative:** `xlm-twitter` (41.4% negative) is substantially more negative than `twitter-roberta` (21.3%) and `bertweet` (11.3%).
- **Most neutral:** `bertweet` (67.4% neutral), followed by `twitter-roberta` (61.3%) and `xlm-twitter` (44.9%).
- **Most positive:** `bertweet` (21.2% positive), then `twitter-roberta` (17.4%), then `xlm-twitter` (13.7%).
- **Highest mean confidence:** `bertweet` (0.766), then `twitter-roberta` (0.692), then `xlm-twitter` (0.539).

The spread across models suggests Moltbook comment sentiment is sensitive to model choice. `xlm-twitter` in particular shifts many comments toward negative relative to the English RoBERTa and BERTweet baselines.

## Per-model artifacts

Each model writes three files:

| Model | Predictions (JSONL) | Predictions (CSV) | Summary (JSON) |
|---|---|---|---|
| `twitter-roberta` | `twitter-roberta_predictions.jsonl` | `twitter-roberta_predictions.csv` | `twitter-roberta_summary.json` |
| `xlm-twitter` | `xlm-twitter_predictions.jsonl` | `xlm-twitter_predictions.csv` | `xlm-twitter_summary.json` |
| `bertweet` | `bertweet_predictions.jsonl` | `bertweet_predictions.csv` | `bertweet_summary.json` |

Combined run metadata: `all_models_summary.json`

## Prediction record schema

Each JSONL row contains:

- `record_id`, `dataset`, `text`, `metadata`
- `transformer_model`, `checkpoint`
- `sentiment_label`, `confidence`
- `probabilities` (`negative`, `neutral`, `positive`)
- `text_chunks` (1 for truncated runs; higher when using `--long-text-strategy window`)

CSV files mirror the same fields with flattened probability columns and `metadata_json`.

## Notes

- These outputs are benchmark predictions only; they do not modify staged Moltbook source data.
- BERTweet truncates long comments to 128 tokens in this run. For fuller coverage of long comments, rerun with `--long-text-strategy window --stride 64`.
- Compare against rule-based ensemble results in `data/rule_based/` and gold-set evaluation in `data/gold/` when assessing model quality.
