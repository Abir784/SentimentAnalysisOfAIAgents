# SentimentAnalysis

## Data Organization

The project uses a structured folder hierarchy for different data processing stages:

```
data/
├── raw/              # Timestamped raw posts (audit trail, never modified)
├── staged/           # Consolidated raw comments (moltbook_comments_all.jsonl - appended on each collection)
├── eda/              # EDA reports and summary statistics
├── preprocessed/     # English-only cleaned text, ready for analysis
└── polarity/         # Polarity-scored comments with sentiment labels
```

## Unified NLP System (Single Command Interface)

Use one orchestrator to run the full research pipeline, selected stages, or specific notebook cells:

```powershell
python scripts/run_nlp_pipeline.py list
```

Run the full script-only pipeline (fast default):

```powershell
python scripts/run_nlp_pipeline.py run --all
```

Run the full research pipeline including notebooks:

```powershell
python scripts/run_nlp_pipeline.py run --all --include-notebooks
```

Run a stage range in pipeline order:

```powershell
python scripts/run_nlp_pipeline.py run --from-stage raw_to_staged --to-stage polarity
```

Run specific stages only:

```powershell
python scripts/run_nlp_pipeline.py run --stages collect,raw_to_staged,polarity,modeling
```

Run one notebook with selected cell numbers:

```powershell
python scripts/run_nlp_pipeline.py run-notebook --notebook notebooks/moltbook_preprocessing_steps.ipynb --cells 1-6,8
```

Run notebook stages with stage-specific cell selectors:

```powershell
python scripts/run_nlp_pipeline.py run --stages preprocess_notebook,polarity_notebook --include-notebooks --notebook-cells preprocess_notebook:1-10 polarity_notebook:1-7
```

Pipeline order is aligned to the research motive (reproducibility, preprocessing robustness, and lightweight model benchmarking):

1. `collect`
2. `raw_to_staged`
3. `eda_summary`
4. `eda_notebook`
5. `preprocess_notebook`
6. `polarity`
7. `polarity_notebook`
8. `modeling`

## Step 1: Collect Data

By default, collection reads URLs from `moltbook_urls.txt` at the repository root.

```powershell
python scripts/run_moltbook_collection.py --config configs/moltbook_collection.url.json
```

Outputs:
- `data/raw/moltbook_raw_<run_id>.jsonl` (timestamped, for audit trail)

Direct URL mode:
```powershell
python scripts/run_moltbook_collection.py --url https://www.moltbook.com/post/<id>
```

Explicit file mode:
```powershell
python scripts/run_moltbook_collection.py --urls-file moltbook_urls.txt
```

## Step 2: Process Raw to Staged

Extract comments from all new/unprocessed raw files and append to the consolidated staged file:

```powershell
python scripts/process_raw_to_staged.py
```

Outputs:
- `data/staged/moltbook_comments_all.jsonl` (consolidated, appended on each run)
- `data/staged/.processed_raw_files.json` (tracking file - prevents reprocessing)

This script:
- Reads all raw JSONL files from `data/raw/`
- Extracts comments from unprocessed files only
- Appends them to `data/staged/moltbook_comments_all.jsonl`
- Tracks processed files to avoid duplication

## Step 3: Run EDA

Analyze the consolidated staged comments:

```powershell
python scripts/run_moltbook_sentiment.py
```

Outputs:
- `data/eda/moltbook_eda_summary_<run_id>.json`

Reads from:
- `data/staged/moltbook_comments_all.jsonl`

## Step 4: Preprocess & Score Polarity

Run the stricter traditional NLP pipeline for English-only preprocessing, side-by-side polarity scoring, and export artifacts:

```powershell
python scripts/run_moltbook_polarity.py
```

Outputs:
- `data/preprocessed/moltbook_comments_preprocessed_<run_id>.jsonl` (cleaned English-only text)
- `data/preprocessed/moltbook_training_ready_<run_id>.csv` (lean modeling dataset)
- `data/polarity/moltbook_comments_polarity_<run_id>.jsonl` (raw vs preprocessed polarity scores)
- `data/polarity/moltbook_polarity_summary_<run_id>.json` (summary statistics and preprocessing policy)

Use the notebooks for interactive inspection and comparison:

```powershell
jupyter notebook notebooks/moltbook_polarity_assessment.ipynb
```

```powershell
jupyter notebook notebooks/moltbook_raw_vs_preprocessed_polarity.ipynb
```

Outputs:
- `data/preprocessed/moltbook_comments_preprocessed_<run_id>.jsonl` (cleaned English-only text)
- `data/preprocessed/moltbook_training_ready_<run_id>.csv` (training-ready cleaned fields)
- `data/polarity/moltbook_comments_polarity_<run_id>.jsonl` (with raw and preprocessed polarity scores)
- `data/polarity/moltbook_polarity_summary_<run_id>.json` (summary statistics)

### Preprocessing Steps
- HTML entity decoding
- Unicode normalization (NFKC)
- Markdown link stripping
- URL removal
- UI glyph removal
- Language detection (English-only filter)
- Spam marker removal
- Duplicate filtering
- POS-aware lemmatization
- Negation scope prefixing
- Stopword removal with negation/intensity exceptions
- Minimum text length validation

## Installation

```powershell
pip install -r requirements.txt
```

## File Structure

- `src/pipelines/collect_moltbook.py`: Data collection orchestration
- `src/pipelines/eda_moltbook.py`: EDA summary generation
- `src/pipelines/normalize_moltbook.py`: Schema normalization
- `src/collectors/moltbook_scraper.py`: Web scraping logic
- `configs/moltbook_collection.url.json`: Collection configuration
- `notebooks/moltbook_polarity_assessment.ipynb`: Interactive preprocessing and polarity scoring
```

Each run prints:
- this-run totals (`raw_count`, `comments_extracted`, etc.)
- dedup stats (`requested_urls`, `scraped_urls`, `skipped_existing_urls`)
- cumulative stored totals (`stored_*_rows_total`, `stored_*_files_total`)

### Outputs
- Raw payload snapshots: `data/raw/moltbook_raw_<run_id>.jsonl`
- Normalized records: `data/staged/moltbook_normalized_<run_id>.jsonl`
- Comments with levels: `data/staged/moltbook_comments_with_levels_<run_id>.jsonl`

### Comment File Schema
- `platform`, `post_id`, `thread_id`, `comment_id`, `parent_id`
- `level`, `author_id`, `relative_time`, `is_verified`, `upvotes`
- `text`, `source_url`, `fetched_at`

### Notes
- For JS-heavy pages, the scraper uses rendered-text fallback to extract post content and comments.
- `level` is currently set to `0` because reply nesting is not exposed in the rendered feed format.

### Canonical Normalized Fields
- `platform`, `post_id`, `thread_id`, `parent_id`
- `author_id`, `author_type`, `author_model`
- `topic`, `text`, `timestamp_utc`
- `upvotes`, `reply_count`, `source_page`, `fetched_at`, `ingested_at`
