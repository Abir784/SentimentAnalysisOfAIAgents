# Synthetic Transformer Sentiment Results

Pretrained transformer sentiment scores for generated multi-model conversation messages.

## Run context

| Field | Value |
|---|---|
| Dataset | `synthetic` |
| Input | `data/sythetic/conversations.jsonl` |
| Messages scored | 6,000 |
| Device | CUDA |
| Long-text strategy | `truncate` (default) |
| Social-text preprocessing | `@user` / `http` normalization applied for all three models |

Generated with:

```powershell
python scripts/run_transformer_sentiment.py synthetic --device cuda --batch-size 16 --models twitter-roberta xlm-twitter
python scripts/run_transformer_sentiment.py synthetic --device cuda --batch-size 16 --models bertweet --max-length 128
```

## Model results summary

| Model | Checkpoint | Rows | Max length | Mean confidence | Negative | Neutral | Positive |
|---|---|---:|---:|---:|---:|---:|---:|
| `twitter-roberta` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 6,000 | 512 | 0.756 | 3,406 (56.8%) | 2,389 (39.8%) | 205 (3.4%) |
| `xlm-twitter` | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 6,000 | 512 | 0.703 | 3,403 (56.7%) | 2,121 (35.4%) | 476 (7.9%) |
| `bertweet` | `finiteautomata/bertweet-base-sentiment-analysis` | 6,000 | 128 | 0.812 | 3,258 (54.3%) | 2,256 (37.6%) | 486 (8.1%) |

## Cross-model comparison

- **Dominant label:** All three models assign **negative** to the largest share of synthetic messages.
- **Most negative:** `twitter-roberta` (56.8%), closely followed by `xlm-twitter` (56.7%) and `bertweet` (54.3%).
- **Most neutral:** `twitter-roberta` (39.8%), then `bertweet` (37.6%), then `xlm-twitter` (35.4%).
- **Most positive:** `bertweet` (8.1%), then `xlm-twitter` (7.9%), then `twitter-roberta` (3.4%).
- **Highest mean confidence:** `bertweet` (0.812), then `twitter-roberta` (0.756), then `xlm-twitter` (0.703).

Synthetic messages are scored substantially more negative than real Moltbook comments under the same transformer models.

## Per-model artifacts

| Model | Predictions (JSONL) | Predictions (CSV) | Summary (JSON) |
|---|---|---|---|
| `twitter-roberta` | `twitter-roberta_predictions.jsonl` | `twitter-roberta_predictions.csv` | `twitter-roberta_summary.json` |
| `xlm-twitter` | `xlm-twitter_predictions.jsonl` | `xlm-twitter_predictions.csv` | `xlm-twitter_summary.json` |
| `bertweet` | `bertweet_predictions.jsonl` | `bertweet_predictions.csv` | `bertweet_summary.json` |

Combined run metadata: `all_models_summary.json`
