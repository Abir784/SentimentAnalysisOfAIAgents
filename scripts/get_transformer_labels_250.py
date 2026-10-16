"""Get transformer sentiment labels for the 250-comment MoltBook gold set.

Reuse-first workflow (per task spec):
  STEP 1: reuse per-comment predictions from the earlier transformer benchmark
          runs (data/transformer_sentiment/moltbook/*_predictions.csv) when they
          cover all 250 comment_ids.
  STEP 2: only for comment_ids missing from those files, run the models with
          the EXACT original benchmark settings (checkpoint, text input,
          max_length, truncation, batch size, label mapping).

Original benchmark provenance (do not change):
  - Script: scripts/run_transformer_sentiment.py -> src/pipelines/transformer_sentiment.py
  - Input: data/staged/moltbook_comments_all.jsonl, field "text" (stripped raw text)
  - Checkpoints:
      twitter-roberta -> cardiffnlp/twitter-roberta-base-sentiment-latest (max_length 512)
      xlm-twitter     -> cardiffnlp/twitter-xlm-roberta-base-sentiment     (max_length 512)
      bertweet        -> finiteautomata/bertweet-base-sentiment-analysis    (max_length 128)
  - Long-text strategy: truncate (text_chunks == 1 for all rows)
  - Batch size: 16, device: cuda in original run (script uses auto -> cuda/mps/cpu)
  - Text preprocessing: preprocess_social_text (@user / http normalization)
    applied in-memory for ALL THREE models (social_text_preprocessing=True);
    stored CSV "text" column is the raw stripped text.
  - Label mapping (checked in MODEL_SPECS, all three): {0: negative, 1: neutral, 2: positive}

Outputs (never overwrites the input xlsx or any existing result):
  1. transformer_labels_250.csv with columns:
     comment_id, twitter_roberta_label, xlm_twitter_label, bertweet_label
  2. MoltBook_Gold_Annotation_250_with_transformers.xlsx — copy of the input
     workbook with Scorers!F:H (twitter_roberta_label, xlm_twitter_label,
     bertweet_label) filled by comment_id via openpyxl, preserving formulas,
     data validation, conditional formatting and sheet order.

Matching is ALWAYS by comment_id, never by row position.
Gold labels are never used for tuning — only for the final agreement report
(if requested separately). This script does not read gold labels at all
except to leave them untouched.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_ALIASES = ["twitter-roberta", "xlm-twitter", "bertweet"]
CSV_COLUMN = {
    "twitter-roberta": "twitter_roberta_label",
    "xlm-twitter": "xlm_twitter_label",
    "bertweet": "bertweet_label",
}
# Exact max_length values from the original benchmark (see data/transformer_sentiment/moltbook/README.md).
ORIGINAL_MAX_LENGTH = {"twitter-roberta": 512, "xlm-twitter": 512, "bertweet": 128}
ORIGINAL_BATCH_SIZE = 16
ORIGINAL_STRATEGY = "truncate"
SEED = 42
ALLOWED_LABELS = ("negative", "neutral", "positive")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def cohen_kappa(y1, y2, n_labels: int = 3) -> float:
    import numpy as np

    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    conf = np.zeros((n_labels, n_labels), dtype=float)
    for a, b in zip(y1, y2):
        conf[a, b] += 1
    n = conf.sum()
    if n == 0:
        return float("nan")
    p0 = np.trace(conf) / n
    pe = (conf.sum(axis=1) * conf.sum(axis=0)).sum() / (n * n)
    if pe >= 1.0:
        return 1.0
    return float((p0 - pe) / (1.0 - pe))


def load_workbook_ids(xlsx_path: Path):
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    if "Annotation" not in wb.sheetnames or "Scorers" not in wb.sheetnames:
        raise ValueError(f"Expected sheets 'Annotation' and 'Scorers', found {wb.sheetnames}")
    ann_rows = list(wb["Annotation"].iter_rows(min_row=2, values_only=True))
    # Annotation header: ['#', 'comment_id', 'text', ...]
    ann_ids = [r[1] for r in ann_rows]
    ann_text = {r[1]: (r[2] if r[2] is not None else "") for r in ann_rows}
    sc_rows = list(wb["Scorers"].iter_rows(min_row=2, values_only=True))
    sc_ids = [r[0] for r in sc_rows]
    wb.close()
    return ann_ids, ann_text, sc_ids


def load_existing_predictions(pred_dir: Path):
    """Return {alias: {record_id: label}}, plus per-alias checkpoint and text."""
    import pandas as pd

    maps: dict[str, dict[str, str]] = {}
    checkpoints: dict[str, str] = {}
    texts: dict[str, dict[str, str]] = {}
    for alias in MODEL_ALIASES:
        csv_path = pred_dir / f"{alias}_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Existing prediction file not found: {csv_path}. "
                "Cannot guess missing values — stopping."
            )
        df = pd.read_csv(csv_path, usecols=["record_id", "text", "sentiment_label", "checkpoint"])
        bad = set(df["sentiment_label"].unique()) - set(ALLOWED_LABELS)
        if bad:
            raise ValueError(f"{csv_path}: unexpected labels {bad}")
        maps[alias] = dict(zip(df["record_id"].astype(str), df["sentiment_label"].astype(str)))
        checkpoints[alias] = str(df["checkpoint"].iloc[0])
        texts[alias] = dict(zip(df["record_id"].astype(str), df["text"].astype(str)))
    return maps, checkpoints, texts


def print_model_specs() -> None:
    from src.pipelines.transformer_sentiment import MODEL_SPECS

    print("Model checkpoints and id2label mapping (from src/pipelines/transformer_sentiment.py):")
    for alias in MODEL_ALIASES:
        spec = MODEL_SPECS[alias]
        print(
            f"  {alias}: checkpoint={spec.checkpoint} "
            f"max_length={ORIGINAL_MAX_LENGTH[alias]} "
            f"id2label={dict(spec.id2label)} "
            f"social_text_preprocessing={spec.social_text_preprocessing}"
        )
        # Never guess LABEL_0/1/2: assert exact expected mapping.
        assert dict(spec.id2label) == {0: "negative", 1: "neutral", 2: "positive"}, (
            f"Unexpected id2label for {alias}: {dict(spec.id2label)}"
        )
    print("  -> canonical lowercase labels: negative, neutral, positive")


def count_truncated(texts: list[str], alias: str, max_length: int) -> int | None:
    """Count texts whose tokenized length (with special tokens, after the
    original social-text preprocessing) exceeds max_length. Returns None if
    the tokenizer cannot be loaded (offline)."""
    try:
        from transformers import AutoTokenizer

        from src.pipelines.transformer_sentiment import MODEL_SPECS, preprocess_social_text
    except ImportError as exc:
        print(f"  [{alias}] truncation check skipped (missing dependency: {exc})")
        return None
    try:
        spec = MODEL_SPECS[alias]
        tok = AutoTokenizer.from_pretrained(spec.checkpoint)
    except Exception as exc:  # e.g. no network / not cached
        print(f"  [{alias}] truncation check skipped (cannot load tokenizer: {exc})")
        return None
    n_trunc = 0
    for t in texts:
        proc = preprocess_social_text(t) if spec.social_text_preprocessing else t
        ids = tok.encode(proc, add_special_tokens=True, truncation=False)
        if len(ids) > max_length:
            n_trunc += 1
    return n_trunc


def run_missing_with_models(missing_ids: list[str], ann_text: dict, device: str) -> dict[str, dict[str, str]]:
    """Run the original pipeline on just the missing comments (exact settings)."""
    from src.pipelines.transformer_sentiment import MODEL_SPECS, run_model

    print(f"Running models on {len(missing_ids)} missing comments (exact original settings)...")
    records = [
        {"record_id": cid, "dataset": "moltbook", "text": str(ann_text[cid]).strip(), "metadata": {}}
        for cid in missing_ids
    ]
    result: dict[str, dict[str, str]] = {a: {} for a in MODEL_ALIASES}
    for alias in MODEL_ALIASES:
        spec = MODEL_SPECS[alias]
        # Check live HF config mapping before mapping outputs.
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(spec.checkpoint)
            print(f"  [{alias}] live HF config id2label: {getattr(cfg, 'id2label', None)}")
        except Exception as exc:
            print(f"  [{alias}] could not fetch live HF config ({exc}); using MODEL_SPECS mapping {dict(spec.id2label)}")
        outputs, _ = run_model(
            records=records,
            model_alias=alias,
            device_name=device,
            batch_size=ORIGINAL_BATCH_SIZE,
            max_length=ORIGINAL_MAX_LENGTH[alias],
            long_text_strategy=ORIGINAL_STRATEGY,
            stride=128,
        )
        for item in outputs:
            label = str(item["sentiment_label"]).strip().lower()
            assert label in ALLOWED_LABELS, f"Unexpected label {label!r} from {alias}"
            result[alias][item["record_id"]] = label
    return result


def spot_check(labels: dict[str, dict[str, str]], ann_text: dict, seed: int, n: int, device: str) -> None:
    """Rerun the models on n random comments and confirm labels match."""
    import random as _random

    from src.pipelines.transformer_sentiment import MODEL_SPECS, run_model

    _random.Random(seed).shuffle(sample_ids := sorted(labels["twitter-roberta"].keys()))
    sample = sample_ids[:n]
    print(f"Spot-check: rerunning all 3 models on {n} random comments (seed={seed})...")
    records = [
        {"record_id": cid, "dataset": "moltbook", "text": str(ann_text[cid]).strip(), "metadata": {}}
        for cid in sample
    ]
    mismatches = 0
    for alias in MODEL_ALIASES:
        spec = MODEL_SPECS[alias]
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(spec.checkpoint)
            print(f"  [{alias}] HF config id2label: {getattr(cfg, 'id2label', None)}")
        except Exception as exc:
            print(f"  [{alias}] live config check skipped ({exc}); MODEL_SPECS id2label={dict(spec.id2label)}")
        outputs, _ = run_model(
            records=records,
            model_alias=alias,
            device_name=device,
            batch_size=ORIGINAL_BATCH_SIZE,
            max_length=ORIGINAL_MAX_LENGTH[alias],
            long_text_strategy=ORIGINAL_STRATEGY,
            stride=128,
        )
        for item in outputs:
            expected = labels[alias][item["record_id"]]
            got = str(item["sentiment_label"]).strip().lower()
            status = "OK" if got == expected else "MISMATCH"
            if got != expected:
                mismatches += 1
            print(f"    {item['record_id'][:18]}... {alias}: reused={expected} rerun={got} [{status}]")
    if mismatches:
        print(f"Spot-check FAILED: {mismatches} mismatches out of {n * len(MODEL_ALIASES)} predictions.")
    else:
        print(f"Spot-check PASSED: all {n * len(MODEL_ALIASES)} rerun labels match reused predictions.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Get transformer labels for the 250-comment gold set (reuse-first).")
    parser.add_argument("--xlsx", type=Path, default=Path("MoltBook_Gold_Annotation_250.xlsx"))
    parser.add_argument("--pred-dir", type=Path, default=Path("data/transformer_sentiment/moltbook"))
    parser.add_argument("--out-csv", type=Path, default=Path("transformer_labels_250.csv"))
    parser.add_argument("--out-xlsx", type=Path, default=Path("MoltBook_Gold_Annotation_250_with_transformers.xlsx"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--spot-check-n", type=int, default=10)
    parser.add_argument("--skip-spot-check", action="store_true", help="Skip model rerun spot-check (needs torch+transformers).")
    parser.add_argument("--skip-truncation", action="store_true", help="Skip tokenizer truncation counts.")
    args = parser.parse_args()

    set_seeds(args.seed)

    if not args.xlsx.exists():
        raise FileNotFoundError(f"Input workbook not found: {args.xlsx}")
    if args.out_xlsx.resolve() == args.xlsx.resolve():
        raise ValueError("Refusing to overwrite the original xlsx. Use a different --out-xlsx.")

    print_model_specs()

    ann_ids, ann_text, sc_ids = load_workbook_ids(args.xlsx)
    print(f"Annotation rows: {len(ann_ids)}, Scorers rows: {len(sc_ids)}")
    assert len(set(ann_ids)) == 250 and len(set(sc_ids)) == 250, "Expected 250 unique comment_ids in each sheet"
    assert set(ann_ids) == set(sc_ids), "Annotation and Scorers comment_id sets differ"

    maps, checkpoints, pred_texts = load_existing_predictions(args.pred_dir)
    print("Existing prediction files reused:")
    for alias in MODEL_ALIASES:
        print(f"  {alias}: {args.pred_dir / f'{alias}_predictions.csv'} (checkpoint={checkpoints[alias]}, text column='text')")

    # Text-input audit: xlsx text must equal staged/prediction text modulo line endings.
    def _norm(s: str) -> str:
        return str(s).replace("\r\n", "\n").strip()

    for alias in MODEL_ALIASES:
        mism = [cid for cid in ann_ids if _norm(ann_text[cid]) != _norm(pred_texts[alias].get(cid, ""))]
        print(f"  [{alias}] text audit vs xlsx: {len(mism)} mismatches (line-ending/whitespace normalized)")
        if mism:
            print(f"    example: {mism[0]}")

    missing = [cid for cid in ann_ids if any(cid not in maps[a] for a in MODEL_ALIASES)]
    if missing:
        print(f"Missing predictions for {len(missing)} comment_ids — running original pipeline on those only.")
        rerun = run_missing_with_models(missing, ann_text, args.device)
        for alias in MODEL_ALIASES:
            maps[alias].update(rerun[alias])
    else:
        print("Coverage: all 250 comment_ids present in all three existing prediction files. No model rerun needed.")

    # Assemble final label dicts keyed by comment_id.
    labels: dict[str, dict[str, str]] = {
        alias: {cid: maps[alias][cid] for cid in sc_ids} for alias in MODEL_ALIASES
    }

    # ---- CHECKS ----
    print("Checks:")
    print(f"  250 unique comment_ids: {len(set(sc_ids)) == 250} (n={len(set(sc_ids))})")
    for alias in MODEL_ALIASES:
        n_missing = sum(1 for cid in sc_ids if not labels[alias].get(cid))
        print(f"  [{alias}] missing predictions: {n_missing}")
        assert n_missing == 0, f"Missing predictions for {alias}"
        for cid, lab in labels[alias].items():
            assert lab in ALLOWED_LABELS and lab == lab.strip() and " " not in lab, f"Bad label {lab!r} for {cid}"
        counts = Counter(labels[alias][cid] for cid in sc_ids)
        print(f"  [{alias}] value counts: {dict(counts)}")

    if not args.skip_truncation:
        print("Truncation audit (tokenized length > max_length, after original @user/http preprocessing):")
        ordered_texts = [str(ann_text[cid]) for cid in sc_ids]
        for alias in MODEL_ALIASES:
            n_trunc = count_truncated(ordered_texts, alias, ORIGINAL_MAX_LENGTH[alias])
            if n_trunc is not None:
                print(f"  [{alias}] truncated (max_length={ORIGINAL_MAX_LENGTH[alias]}): {n_trunc}/250")
    else:
        print("Truncation audit skipped by flag.")

    # Pairwise agreement + Cohen's kappa.
    import numpy as np

    label_to_id = {lab: i for i, lab in enumerate(ALLOWED_LABELS)}
    print("Pairwise agreement on these 250:")
    pairs = [("twitter-roberta", "xlm-twitter"), ("twitter-roberta", "bertweet"), ("xlm-twitter", "bertweet")]
    for a, b in pairs:
        ya = np.array([label_to_id[labels[a][cid]] for cid in sc_ids])
        yb = np.array([label_to_id[labels[b][cid]] for cid in sc_ids])
        agree = float((ya == yb).mean())
        kappa = cohen_kappa(ya, yb)
        print(f"  {a} vs {b}: agreement={agree:.4f} ({int((ya == yb).sum())}/250), Cohen's kappa={kappa:.4f}")

    # ---- OUTPUT 1: CSV ----
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["comment_id", "twitter_roberta_label", "xlm_twitter_label", "bertweet_label"]
        )
        writer.writeheader()
        for cid in sc_ids:  # Scorers sheet order; matched strictly by comment_id
            writer.writerow(
                {
                    "comment_id": cid,
                    "twitter_roberta_label": labels["twitter-roberta"][cid],
                    "xlm_twitter_label": labels["xlm-twitter"][cid],
                    "bertweet_label": labels["bertweet"][cid],
                }
            )
    print(f"Wrote {args.out_csv} (250 rows, Scorers sheet order)")

    # ---- OUTPUT 2: xlsx copy (openpyxl, preserve everything) ----
    import openpyxl

    wb = openpyxl.load_workbook(args.xlsx)  # data_only=False: keep formulas
    assert wb.sheetnames == ["Guidelines", "Annotation", "Scorers", "Evaluation"], (
        f"Unexpected sheet order: {wb.sheetnames}"
    )
    ws = wb["Scorers"]
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    assert header[0] == "comment_id", f"Unexpected Scorers header: {header[:8]}"
    col_idx = {name: header.index(name) + 1 for name in ["twitter_roberta_label", "xlm_twitter_label", "bertweet_label"]}
    alias_by_colname = {"twitter_roberta_label": "twitter-roberta", "xlm_twitter_label": "xlm-twitter", "bertweet_label": "bertweet"}
    n_filled = 0
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        cid = row[0].value
        if cid is None:
            continue
        cid = str(cid)
        for col_name, j in col_idx.items():
            row[j - 1].value = labels[alias_by_colname[col_name]][cid]
            n_filled += 1
    wb.save(args.out_xlsx)
    print(f"Wrote {args.out_xlsx} (filled {n_filled} label cells on Scorers!F:H by comment_id; all other sheets/cells untouched)")
    wb.close()

    # ---- Spot-check ----
    if not args.skip_spot_check:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            print(f"Spot-check SKIPPED: missing dependency ({exc}). Install requirements_transformers.txt to enable it.")
        else:
            spot_check(labels, ann_text, args.seed, args.spot_check_n, args.device)
    else:
        print("Spot-check skipped by flag.")

    print("Assumptions:")
    print("  1. Existing benchmark predictions fully cover the 250 gold comment_ids, so no new inference was needed for the outputs.")
    print("  2. Original text input = stripped raw staged text (data/staged/moltbook_comments_all.jsonl field 'text');")
    print("     @user/http normalization is applied in-memory at inference time for all three models, not stored.")
    print("  3. Original settings kept exactly: max_length 512/512/128, truncate, batch_size 16, label map {0: negative, 1: neutral, 2: positive}.")
    print("  4. CSV row order follows the Scorers sheet order; xlsx copy fills Scorers!F:H strictly by comment_id.")
    print("  5. Device preference: --device auto (CUDA > MPS > CPU); seeds set for reproducibility; gold labels never used for tuning.")


if __name__ == "__main__":
    main()
