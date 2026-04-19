# SentimentAnalysis

## Rule-Based Pipeline Overview

This repository now uses a rule-based sentiment pipeline with a separate interaction-network stage for RQ1.

Pipeline stages:

1. `data_acquisition` -> `scripts/process_raw_to_staged.py`
2. `text_preprocessing` -> `scripts/run_moltbook_text_preprocessing.py`
3. `eda` -> `scripts/run_moltbook_eda_stage.py`
4. `feature_extraction` -> `scripts/run_moltbook_feature_extraction.py`
5. `rule_based_tools` -> `scripts/run_moltbook_rule_based.py`
6. `interaction_network` (separate RQ1 run) -> `scripts/run_moltbook_interaction_network.py`

There is no ML sentiment training stage in the active pipeline.

## Data Layout

```text
data/
├── raw/                     # Source snapshots (preserved)
├── staged/                  # Consolidated staged comments
├── preprocessed_rule_based/ # Stage 2 outputs
├── eda_rule_based/          # Stage 3 outputs
├── features_rule_based/     # Stage 4 outputs
├── rule_based/              # Stage 5 outputs (VADER/SentiWordNet/Ensemble)
└── eda/                     # RQ1 interaction-network artifacts
```

## Unified Orchestrator

List stages:

```powershell
python scripts/run_nlp_pipeline.py list
```

Run full pipeline (script stages):

```powershell
python scripts/run_nlp_pipeline.py run --all
```

Run selected stages:

```powershell
python scripts/run_nlp_pipeline.py run --stages data_acquisition,text_preprocessing,eda,feature_extraction,rule_based_tools,interaction_network
```

Run a contiguous range:

```powershell
python scripts/run_nlp_pipeline.py run --from-stage text_preprocessing --to-stage rule_based_tools
```

## Individual Stage Commands

```powershell
python scripts/process_raw_to_staged.py
python scripts/run_moltbook_text_preprocessing.py
python scripts/run_moltbook_eda_stage.py
python scripts/run_moltbook_feature_extraction.py
python scripts/run_moltbook_rule_based.py
python scripts/run_moltbook_interaction_network.py
```

Interaction network edge modes:

- `--edge-mode auto` (default)
- `--edge-mode direct`
- `--edge-mode sequential`

## Dashboard

Launch Streamlit dashboard:

```powershell
streamlit run dashboard/app.py
```

Dashboard tabs:

- Overview
- Rule-Based Results
- Feature Extraction
- RQ1 Analysis (separate interaction network)

## Install

```powershell
pip install -r requirements.txt
```

## Notes

- Raw data is the permanent source of truth.
- Regenerate downstream folders by rerunning the pipeline after cleanup.
- Custom model algorithm history is archived in `custom_model_algorithm.txt` for record keeping only.
