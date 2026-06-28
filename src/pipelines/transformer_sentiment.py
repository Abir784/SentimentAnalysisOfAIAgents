from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


CANONICAL_LABELS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    id2label: Mapping[int, str]
    social_text_preprocessing: bool = True


MODEL_SPECS: dict[str, ModelSpec] = {
    "twitter-roberta": ModelSpec(
        checkpoint="cardiffnlp/twitter-roberta-base-sentiment-latest",
        id2label={0: "negative", 1: "neutral", 2: "positive"},
    ),
    "xlm-twitter": ModelSpec(
        checkpoint="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        id2label={0: "negative", 1: "neutral", 2: "positive"},
    ),
    "bertweet": ModelSpec(
        checkpoint="finiteautomata/bertweet-base-sentiment-analysis",
        id2label={0: "negative", 1: "neutral", 2: "positive"},
    ),
    "distilbert-sst2": ModelSpec(
        checkpoint="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        id2label={0: "negative", 1: "positive"},
        social_text_preprocessing=False,
    ),
}


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}")
            yield line_number, value


def load_moltbook_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, item in read_jsonl(path):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        records.append(
            {
                "record_id": str(item.get("comment_id") or f"line-{line_number}"),
                "dataset": "moltbook",
                "text": text,
                "metadata": {
                    "post_id": item.get("post_id"),
                    "thread_id": item.get("thread_id"),
                    "comment_id": item.get("comment_id"),
                    "parent_id": item.get("parent_id"),
                    "author_id": item.get("author_id"),
                    "level": item.get("level"),
                    "is_verified": item.get("is_verified"),
                    "upvotes": item.get("upvotes"),
                    "source_url": item.get("source_url"),
                    "fetched_at": item.get("fetched_at"),
                },
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def load_synthetic_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, conversation in read_jsonl(path):
        messages = conversation.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError(f"Expected 'messages' to be a list on line {line_number} of {path}")

        for message_index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                continue
            text = str(message.get("message") or "").strip()
            if not text:
                continue
            records.append(
                {
                    "record_id": f"conversation-{line_number}-message-{message_index}",
                    "dataset": "synthetic",
                    "text": text,
                    "metadata": {
                        "conversation_line": line_number,
                        "message_index": message_index,
                        "timestamp": conversation.get("timestamp"),
                        "topic": conversation.get("topic"),
                        "model_a": conversation.get("model_a"),
                        "model_b": conversation.get("model_b"),
                        "turn_count": conversation.get("turn_count"),
                        "turn": message.get("turn"),
                        "speaker": message.get("speaker"),
                    },
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def preprocess_social_text(text: str) -> str:
    parts = []
    for token in text.split():
        if token.startswith("@") and len(token) > 1:
            parts.append("@user")
        elif re.match(r"https?://", token, flags=re.IGNORECASE):
            parts.append("http")
        else:
            parts.append(token)
    return " ".join(parts)


def resolve_device(torch: Any, requested: str) -> tuple[Any, str]:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is not available.")
    return torch.device(requested), requested


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _normalize_probabilities(
    probabilities: Sequence[float], id2label: Mapping[int, str]
) -> dict[str, float]:
    result = {label: 0.0 for label in CANONICAL_LABELS}
    for class_id, probability in enumerate(probabilities):
        label = id2label.get(class_id)
        if label in result:
            result[label] += float(probability)
    return result


def _batched(values: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _predict_truncated(
    texts: Sequence[str], tokenizer: Any, model: Any, torch: Any, device: Any,
    batch_size: int, max_length: int,
) -> list[tuple[list[float], int]]:
    predictions: list[tuple[list[float], int]] = []
    for batch_texts in _batched(texts, batch_size):
        encoded = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**_move_batch(encoded, device)).logits
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        predictions.extend((row, 1) for row in probabilities)
    return predictions


def _predict_windowed(
    texts: Sequence[str], tokenizer: Any, model: Any, torch: Any, device: Any,
    batch_size: int, max_length: int, stride: int,
) -> list[tuple[list[float], int]]:
    predictions: list[tuple[list[float], int]] = []
    for text in texts:
        encoded = tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
        )
        windows = [
            {key: encoded[key][index] for key in ("input_ids", "attention_mask")}
            for index in range(len(encoded["input_ids"]))
        ]
        window_probabilities: list[Any] = []
        for window_batch in _batched(windows, batch_size):
            padded = tokenizer.pad(list(window_batch), padding=True, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**_move_batch(padded, device)).logits
                window_probabilities.append(torch.softmax(logits, dim=-1).detach().cpu())
        mean_probability = torch.cat(window_probabilities, dim=0).mean(dim=0).tolist()
        predictions.append((mean_probability, len(windows)))
    return predictions


def run_model(
    records: Sequence[dict[str, Any]], model_alias: str, device_name: str = "auto",
    batch_size: int = 16, max_length: int = 512, long_text_strategy: str = "truncate",
    stride: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer dependencies are missing. Install requirements_transformers.txt first."
        ) from exc

    spec = MODEL_SPECS[model_alias]
    device, resolved_device = resolve_device(torch, device_name)
    tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(spec.checkpoint)
    model.to(device)
    model.eval()

    texts = [
        preprocess_social_text(record["text"])
        if spec.social_text_preprocessing
        else record["text"]
        for record in records
    ]
    if long_text_strategy == "window":
        raw_predictions = _predict_windowed(
            texts, tokenizer, model, torch, device, batch_size, max_length, stride
        )
    else:
        raw_predictions = _predict_truncated(
            texts, tokenizer, model, torch, device, batch_size, max_length
        )

    outputs: list[dict[str, Any]] = []
    for record, (raw_probabilities, chunk_count) in zip(records, raw_predictions):
        probabilities = _normalize_probabilities(raw_probabilities, spec.id2label)
        label = max(probabilities, key=probabilities.get)
        outputs.append(
            {
                **record,
                "transformer_model": model_alias,
                "checkpoint": spec.checkpoint,
                "sentiment_label": label,
                "confidence": round(probabilities[label], 8),
                "probabilities": {
                    key: round(value, 8) for key, value in probabilities.items()
                },
                "supported_labels": sorted(set(spec.id2label.values())),
                "text_chunks": chunk_count,
            }
        )

    counts = Counter(item["sentiment_label"] for item in outputs)
    summary = {
        "dataset": records[0]["dataset"] if records else None,
        "model_alias": model_alias,
        "checkpoint": spec.checkpoint,
        "device": resolved_device,
        "rows_scored": len(outputs),
        "supported_labels": sorted(set(spec.id2label.values())),
        "long_text_strategy": long_text_strategy,
        "max_length": max_length,
        "stride": stride if long_text_strategy == "window" else None,
        "label_counts": {label: counts.get(label, 0) for label in CANONICAL_LABELS},
        "label_shares": {
            label: round(counts.get(label, 0) / len(outputs), 6) if outputs else 0.0
            for label in CANONICAL_LABELS
        },
        "mean_confidence": round(
            sum(item["confidence"] for item in outputs) / len(outputs)
            if outputs
            else 0.0,
            6,
        ),
    }
    return outputs, summary


def write_results(
    outputs: Sequence[dict[str, Any]], summary: dict[str, Any], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    alias = summary["model_alias"]
    jsonl_path = output_dir / f"{alias}_predictions.jsonl"
    csv_path = output_dir / f"{alias}_predictions.csv"
    summary_path = output_dir / f"{alias}_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in outputs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    csv_fields = [
        "record_id", "dataset", "text", "transformer_model", "checkpoint",
        "sentiment_label", "confidence", "negative_probability",
        "neutral_probability", "positive_probability", "text_chunks", "metadata_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in outputs:
            probabilities = item["probabilities"]
            writer.writerow(
                {
                    "record_id": item["record_id"],
                    "dataset": item["dataset"],
                    "text": item["text"],
                    "transformer_model": item["transformer_model"],
                    "checkpoint": item["checkpoint"],
                    "sentiment_label": item["sentiment_label"],
                    "confidence": item["confidence"],
                    "negative_probability": probabilities["negative"],
                    "neutral_probability": probabilities["neutral"],
                    "positive_probability": probabilities["positive"],
                    "text_chunks": item["text_chunks"],
                    "metadata_json": json.dumps(item["metadata"], ensure_ascii=False),
                }
            )

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"jsonl": jsonl_path, "csv": csv_path, "summary": summary_path}


def validate_run_options(
    records: Sequence[dict[str, Any]], batch_size: int, max_length: int,
    long_text_strategy: str, stride: int,
) -> None:
    if not records:
        raise ValueError("The input contained no non-empty text records.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if max_length < 8:
        raise ValueError("max_length must be at least 8.")
    if long_text_strategy == "window" and not 0 <= stride < max_length:
        raise ValueError("stride must be non-negative and smaller than max_length.")


def run_models(
    records: Sequence[dict[str, Any]], model_aliases: Iterable[str], output_dir: Path,
    device: str, batch_size: int, max_length: int, long_text_strategy: str, stride: int,
) -> list[dict[str, Any]]:
    validate_run_options(records, batch_size, max_length, long_text_strategy, stride)
    run_summaries = []
    for model_alias in model_aliases:
        print(f"Loading {model_alias}: {MODEL_SPECS[model_alias].checkpoint}")
        outputs, summary = run_model(
            records=records,
            model_alias=model_alias,
            device_name=device,
            batch_size=batch_size,
            max_length=max_length,
            long_text_strategy=long_text_strategy,
            stride=stride,
        )
        paths = write_results(outputs, summary, output_dir)
        summary["artifacts"] = {key: path.as_posix() for key, path in paths.items()}
        paths["summary"].write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        run_summaries.append(summary)
        print(f"Scored {len(outputs):,} records with {model_alias}")
        print(f"Predictions: {paths['jsonl']}")
    return run_summaries
