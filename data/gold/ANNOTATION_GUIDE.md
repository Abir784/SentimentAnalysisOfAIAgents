# MoltBook Gold-Set Annotation Guide

Purpose: create a two-rater sentiment gold set for evaluation and calibration.

## Input File

Use the sampled file:

- `data/gold/moltbook_goldset_sample_<RUN_ID>.csv`

Do not edit any metadata columns. Only fill:

- `rater_1_label`
- `rater_2_label`
- `adjudicated_label`
- `adjudication_notes`

Allowed labels:

- `negative`
- `neutral`
- `positive`

## Labeling Rules

- `positive`: text expresses approval, support, optimistic or favorable affect.
- `negative`: text expresses disapproval, critique, frustration, pessimism, or adverse affect.
- `neutral`: descriptive/informational text with no clear positive or negative valence.

Tie-break protocol:

1. Rater 1 and Rater 2 label independently.
2. If labels match, copy that label to `adjudicated_label`.
3. If labels differ, discuss and set final `adjudicated_label`.
4. Record rationale briefly in `adjudication_notes`.

## Evaluation Command

After annotation is complete:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_moltbook_goldset.py --input data\gold\moltbook_goldset_sample_<RUN_ID>.csv
```

This writes:

- `data/gold/moltbook_goldset_sample_<RUN_ID>_evaluation.json`

Reported outputs include:

- Cohen kappa between raters
- Macro-F1 for VADER, SentiWordNet, and Ensemble against adjudicated human labels
