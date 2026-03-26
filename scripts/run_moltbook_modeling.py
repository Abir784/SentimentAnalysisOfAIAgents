from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from time import perf_counter


def _find_latest_training_csv(explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Training file not found: {path}")
        return path

    candidates = sorted(Path("data/preprocessed").glob("moltbook_training_ready_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No training CSV found under data/preprocessed. Run scripts/run_moltbook_polarity.py first."
        )
    return candidates[-1]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }


def _predict_with_thresholds(
    probas: np.ndarray,
    classes: np.ndarray,
    thresholds: Dict[str, float],
) -> np.ndarray:
    ratio = np.ones_like(probas)
    for i, cls in enumerate(classes):
        ratio[:, i] = probas[:, i] / max(thresholds.get(str(cls), 0.5), 1e-6)
    return classes[np.argmax(ratio, axis=1)]


def _minority_threshold_search(
    probas: np.ndarray,
    classes: np.ndarray,
    y_true: np.ndarray,
    majority_label: str,
    threshold_grid: List[float],
) -> Dict[str, float]:
    thresholds = {str(c): 0.5 for c in classes}
    class_to_index = {str(c): i for i, c in enumerate(classes)}

    for cls in classes:
        cls_str = str(cls)
        if cls_str == majority_label:
            continue
        idx = class_to_index[cls_str]
        y_true_bin = (y_true == cls_str).astype(int)

        best_f1 = -1.0
        best_t = 0.5
        for t in threshold_grid:
            pred_bin = (probas[:, idx] >= t).astype(int)
            f1_val = f1_score(y_true_bin, pred_bin, zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_t = t
        thresholds[cls_str] = float(best_t)

    return thresholds


def _build_model(model_name: str):
    if model_name == "logistic_lr":
        base = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if model_name == "linear_svm":
        return LinearSVC(class_weight="balanced", random_state=42)
    if model_name == "sgd_linear":
        return SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, random_state=42)
    if model_name == "naive_bayes":
        return MultinomialNB(alpha=0.5)
    raise ValueError(f"Unsupported model: {model_name}")


def _resolve_text_series(df: pd.DataFrame, preferred_col: str, fallback_col: str) -> pd.Series:
    if preferred_col in df.columns:
        return df[preferred_col].fillna("").astype(str)
    if fallback_col in df.columns:
        return df[fallback_col].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def _neutral_guard_threshold(train_max_proba: np.ndarray, y_train: np.ndarray) -> float:
    neutral_mask = y_train == "neutral"
    if neutral_mask.sum() == 0:
        return 0.45

    neutral_probs = train_max_proba[neutral_mask]
    if len(neutral_probs) == 0:
        return 0.45

    return float(np.clip(np.quantile(neutral_probs, 0.7), 0.34, 0.62))


def _run_moltbook_dualview_resonance_oof(
    model_df: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y_arr = y.astype(str).values
    oof_pred = np.empty(len(y_arr), dtype=object)

    fold_acc: List[float] = []
    fold_f1: List[float] = []
    fold_time_sec: List[float] = []
    neutral_thresholds: List[float] = []

    basic_text = _resolve_text_series(model_df, "text_basic_clean", "text_traditional_clean")
    trad_text = _resolve_text_series(model_df, "text_traditional_clean", "text_basic_clean")
    text_len_words = pd.to_numeric(
        model_df.get("text_len_words_traditional_clean", pd.Series([0] * len(model_df))),
        errors="coerce",
    ).fillna(0.0)
    abs_delta = pd.to_numeric(
        model_df.get("polarity_compound_delta", pd.Series([0.0] * len(model_df))),
        errors="coerce",
    ).fillna(0.0).abs()

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(trad_text, y_arr), start=1):
        t0 = perf_counter()
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]

        trad_train = trad_text.iloc[train_idx]
        trad_test = trad_text.iloc[test_idx]
        basic_train = basic_text.iloc[train_idx]
        basic_test = basic_text.iloc[test_idx]

        # Two lexical views + character morphology view.
        vec_word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=3500,
        )
        vec_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            max_features=3000,
        )

        x_word_train = vec_word.fit_transform(trad_train)
        x_word_test = vec_word.transform(trad_test)
        x_char_train = vec_char.fit_transform(basic_train)
        x_char_test = vec_char.transform(basic_test)

        base_word = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=2500,
            random_state=42,
        )
        base_char = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )
        base_hybrid = MultinomialNB(alpha=0.35)

        x_hybrid_train = hstack([x_word_train, x_char_train])
        x_hybrid_test = hstack([x_word_test, x_char_test])

        base_word.fit(x_word_train, y_train)
        base_char.fit(x_char_train, y_train)
        base_hybrid.fit(x_hybrid_train, y_train)

        classes = np.array(sorted(list(set(y_train))))

        p_word_train = base_word.predict_proba(x_word_train)
        p_char_train = base_char.predict_proba(x_char_train)
        p_hybrid_train = base_hybrid.predict_proba(x_hybrid_train)

        p_word_test = base_word.predict_proba(x_word_test)
        p_char_test = base_char.predict_proba(x_char_test)
        p_hybrid_test = base_hybrid.predict_proba(x_hybrid_test)

        max_word_train = np.max(p_word_train, axis=1)
        max_char_train = np.max(p_char_train, axis=1)
        max_hybrid_train = np.max(p_hybrid_train, axis=1)

        max_word_test = np.max(p_word_test, axis=1)
        max_char_test = np.max(p_char_test, axis=1)
        max_hybrid_test = np.max(p_hybrid_test, axis=1)

        consensus_train = np.abs(max_word_train - max_char_train)
        consensus_test = np.abs(max_word_test - max_char_test)

        train_meta = np.column_stack(
            [
                p_word_train,
                p_char_train,
                p_hybrid_train,
                consensus_train,
                abs_delta.iloc[train_idx].values,
                text_len_words.iloc[train_idx].values,
            ]
        )
        test_meta = np.column_stack(
            [
                p_word_test,
                p_char_test,
                p_hybrid_test,
                consensus_test,
                abs_delta.iloc[test_idx].values,
                text_len_words.iloc[test_idx].values,
            ]
        )

        meta = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )
        meta.fit(train_meta, y_train)

        train_meta_proba = meta.predict_proba(train_meta)
        test_meta_proba = meta.predict_proba(test_meta)
        max_meta_train = np.max(train_meta_proba, axis=1)
        max_meta_test = np.max(test_meta_proba, axis=1)

        neutral_threshold = _neutral_guard_threshold(max_meta_train, y_train)
        neutral_thresholds.append(neutral_threshold)

        raw_pred = meta.classes_[np.argmax(test_meta_proba, axis=1)]
        force_neutral = (max_meta_test < neutral_threshold) & (consensus_test < 0.22)
        fold_pred = raw_pred.astype(object)
        fold_pred[force_neutral] = "neutral"

        oof_pred[test_idx] = fold_pred
        fold_acc.append(float(accuracy_score(y_test, fold_pred)))
        fold_f1.append(float(f1_score(y_test, fold_pred, average="macro", zero_division=0)))
        fold_time_sec.append(float(perf_counter() - t0))

        print(
            f"  moltbook_dualview_resonance fold={fold_id} "
            f"acc={fold_acc[-1]:.4f} f1_macro={fold_f1[-1]:.4f} "
            f"neutral_guard={neutral_threshold:.3f}"
        )

    extra: Dict[str, Any] = {
        "cv_accuracy_std": float(np.std(np.array(fold_acc), ddof=0)),
        "cv_f1_macro_std": float(np.std(np.array(fold_f1), ddof=0)),
        "runtime_mean_sec": float(np.mean(np.array(fold_time_sec))),
        "neutral_guard_threshold_mean": float(np.mean(np.array(neutral_thresholds))),
        "model_notes": (
            "Custom dual-view resonance stack: word TF-IDF + char TF-IDF + hybrid NB with "
            "confidence-consensus neutral guard for minority stabilization."
        ),
    }
    return oof_pred, extra


def _run_model_oof(
    model_name: str,
    x_text: pd.Series,
    y: pd.Series,
    cv: StratifiedKFold,
    threshold_grid: List[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y_arr = y.astype(str).values
    oof_pred = np.empty(len(y_arr), dtype=object)

    fold_acc: List[float] = []
    fold_f1: List[float] = []
    fold_time_sec: List[float] = []
    selected_thresholds: List[Dict[str, float]] = []

    majority_label = y.value_counts().idxmax()

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(x_text, y_arr), start=1):
        t0 = perf_counter()
        x_train = x_text.iloc[train_idx]
        x_test = x_text.iloc[test_idx]
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=4000,
        )
        x_train_vec = vectorizer.fit_transform(x_train)
        x_test_vec = vectorizer.transform(x_test)

        model = _build_model(model_name)
        model.fit(x_train_vec, y_train)

        if hasattr(model, "predict_proba") and model_name in {"logistic_lr", "sgd_linear"}:
            train_proba = model.predict_proba(x_train_vec)
            thresholds = _minority_threshold_search(
                probas=train_proba,
                classes=model.classes_,
                y_true=y_train,
                majority_label=str(majority_label),
                threshold_grid=threshold_grid,
            )
            test_proba = model.predict_proba(x_test_vec)
            fold_pred = _predict_with_thresholds(test_proba, model.classes_, thresholds)
            selected_thresholds.append(thresholds)
        else:
            fold_pred = model.predict(x_test_vec)

        oof_pred[test_idx] = fold_pred
        fold_acc.append(float(accuracy_score(y_test, fold_pred)))
        fold_f1.append(float(f1_score(y_test, fold_pred, average="macro", zero_division=0)))
        fold_time_sec.append(float(perf_counter() - t0))

        print(f"  {model_name} fold={fold_id} acc={fold_acc[-1]:.4f} f1_macro={fold_f1[-1]:.4f}")

    extra: Dict[str, Any] = {
        "cv_accuracy_std": float(np.std(np.array(fold_acc), ddof=0)),
        "cv_f1_macro_std": float(np.std(np.array(fold_f1), ddof=0)),
        "runtime_mean_sec": float(np.mean(np.array(fold_time_sec))),
    }
    if selected_thresholds:
        avg_thresholds: Dict[str, float] = {}
        keys = sorted(selected_thresholds[0].keys())
        for key in keys:
            avg_thresholds[key] = float(np.mean([d.get(key, 0.5) for d in selected_thresholds]))
        extra["avg_thresholds"] = avg_thresholds

    return oof_pred, extra


def _plot_metric_bars(metrics_by_model: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    model_names = list(metrics_by_model.keys())
    acc = [metrics_by_model[m]["accuracy"] for m in model_names]
    f1m = [metrics_by_model[m]["f1_macro"] for m in model_names]
    acc_std = [metrics_by_model[m].get("cv_accuracy_std", 0.0) for m in model_names]
    f1_std = [metrics_by_model[m].get("cv_f1_macro_std", 0.0) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, acc, width=width, yerr=acc_std, capsize=4, label="Accuracy")
    plt.bar(x + width / 2, f1m, width=width, yerr=f1_std, capsize=4, label="Macro F1")
    plt.xticks(x, model_names, rotation=10)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Lightweight ML Models: Accuracy vs Macro F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_requested_metrics(metrics_by_model: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    model_names = list(metrics_by_model.keys())
    metric_names = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "sustainability"]
    metric_titles = ["Accuracy", "F1 Score", "Precision", "Recall", "Sustainability"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for i, (metric_name, metric_title) in enumerate(zip(metric_names, metric_titles)):
        ax = axes_flat[i]
        values = [metrics_by_model[m].get(metric_name, 0.0) for m in model_names]
        sns.barplot(x=model_names, y=values, ax=ax, palette="Blues_d")
        ax.set_title(metric_title)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        for j, v in enumerate(values):
            ax.text(j, min(v + 0.02, 0.98), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes_flat[-1].axis("off")
    plt.suptitle("Requested Evaluation Metrics by Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_confusion_matrices(
    metrics_by_model: Dict[str, Dict[str, Any]], labels: List[str], out_path: Path
) -> None:
    model_names = list(metrics_by_model.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)

    for idx, model_name in enumerate(model_names):
        ax = axes[0, idx]
        cm = np.array(metrics_by_model[model_name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_classwise_f1(metrics_by_model: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for model_name, model_metrics in metrics_by_model.items():
        report = model_metrics.get("classification_report", {})
        for label, values in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            if isinstance(values, dict) and "f1-score" in values:
                rows.append({"model": model_name, "label": label, "f1": values["f1-score"]})

    if not rows:
        return

    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_df, x="label", y="f1", hue="model")
    plt.ylim(0, 1)
    plt.title("Class-wise F1 by Lightweight Model")
    plt.ylabel("F1 Score")
    plt.xlabel("Polarity Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate lightweight ML sentiment models for MoltBook comments."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Path to training CSV. Default: latest data/preprocessed/moltbook_training_ready_*.csv",
    )
    parser.add_argument(
        "--label-col",
        default="processed_polarity_label",
        help="Target label column in the training file.",
    )
    parser.add_argument(
        "--text-col",
        default="text_traditional_clean",
        help="Text feature column for modeling.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified K-fold count.")
    parser.add_argument(
        "--threshold-grid",
        default="0.25,0.3,0.35,0.4,0.45,0.5",
        help="Thresholds for minority tuning on probabilistic models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "logistic_lr",
            "linear_svm",
            "sgd_linear",
            "naive_bayes",
            "moltbook_dualview_resonance",
        ],
        help="Lightweight models to run.",
    )
    args = parser.parse_args()

    if len(args.models) < 3:
        raise ValueError("Specify at least three lightweight models in --models.")

    input_path = _find_latest_training_csv(args.input)
    df = pd.read_csv(input_path)

    required_cols = [args.text_col, args.label_col, "comment_id"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    optional_cols = [
        "text_basic_clean",
        "text_traditional_clean",
        "text_len_words_traditional_clean",
        "polarity_compound_delta",
    ]
    keep_cols = ["comment_id", args.text_col, args.label_col] + [
        c for c in optional_cols if c in df.columns and c not in {"comment_id", args.text_col, args.label_col}
    ]

    model_df = df[keep_cols].copy()
    model_df = model_df.dropna(subset=[args.text_col, args.label_col])
    model_df[args.text_col] = model_df[args.text_col].astype(str)
    model_df[args.label_col] = model_df[args.label_col].astype(str)

    y_all = model_df[args.label_col]
    min_class_count = int(y_all.value_counts().min())
    n_splits = min(args.cv_folds, max(2, min_class_count))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    thresholds = [float(x.strip()) for x in args.threshold_grid.split(",") if x.strip()]

    x_all = model_df[args.text_col]
    ids_all = model_df["comment_id"].astype(str)
    labels = sorted(y_all.unique().tolist())

    preds_by_model: Dict[str, np.ndarray] = {}
    metrics_by_model: Dict[str, Dict[str, Any]] = {}

    print(f"Modeling input: {input_path}")
    print(f"Rows: {len(model_df)} | CV folds: {n_splits} | Labels: {labels}")

    for model_name in args.models:
        if model_name == "moltbook_dualview_resonance":
            preds, extra = _run_moltbook_dualview_resonance_oof(model_df, y_all, cv)
        else:
            preds, extra = _run_model_oof(model_name, x_all, y_all, cv, thresholds)
        m = _metrics(y_all.values, preds, labels)
        m.update(extra)
        preds_by_model[model_name] = preds
        metrics_by_model[model_name] = m

    runtimes = [metrics_by_model[m].get("runtime_mean_sec", 0.0) for m in metrics_by_model]
    rt_min = min(runtimes)
    rt_max = max(runtimes)
    for model_name in metrics_by_model:
        runtime = metrics_by_model[model_name].get("runtime_mean_sec", 0.0)
        if rt_max > rt_min:
            sustainability = (rt_max - runtime) / (rt_max - rt_min)
        else:
            sustainability = 1.0
        metrics_by_model[model_name]["sustainability"] = float(sustainability)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/modeling")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({"comment_id": ids_all.values, "y_true": y_all.values, "text": x_all.values})
    for model_name, preds in preds_by_model.items():
        pred_df[f"pred_{model_name}"] = preds

    pred_path = out_dir / f"moltbook_model_predictions_{run_id}.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    metric_plot_path = out_dir / f"moltbook_model_metrics_bar_{run_id}.png"
    requested_metrics_plot_path = out_dir / f"moltbook_model_requested_metrics_{run_id}.png"
    confusion_plot_path = out_dir / f"moltbook_model_confusion_matrices_{run_id}.png"
    classwise_plot_path = out_dir / f"moltbook_model_classwise_f1_{run_id}.png"

    _plot_metric_bars(metrics_by_model, metric_plot_path)
    _plot_requested_metrics(metrics_by_model, requested_metrics_plot_path)
    _plot_confusion_matrices(metrics_by_model, labels, confusion_plot_path)
    _plot_classwise_f1(metrics_by_model, classwise_plot_path)

    summary = {
        "run_id": run_id,
        "input_path": str(input_path).replace("\\", "/"),
        "rows_used": int(len(model_df)),
        "cv_folds": int(n_splits),
        "label_col": args.label_col,
        "text_col": args.text_col,
        "models": metrics_by_model,
        "predictions_path": str(pred_path).replace("\\", "/"),
        "plots": {
            "metrics_bar": str(metric_plot_path).replace("\\", "/"),
            "requested_metrics": str(requested_metrics_plot_path).replace("\\", "/"),
            "confusion_matrices": str(confusion_plot_path).replace("\\", "/"),
            "classwise_f1": str(classwise_plot_path).replace("\\", "/"),
        },
    }
    summary_path = out_dir / f"moltbook_model_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print("Modeling complete")
    print(f"input_file: {input_path}")
    print(f"rows_used: {len(model_df)}")
    print(f"cv_folds: {n_splits}")
    for model_name, model_metrics in metrics_by_model.items():
        print(f"{model_name}_accuracy: {model_metrics['accuracy']:.4f}")
        print(f"{model_name}_f1_macro: {model_metrics['f1_macro']:.4f}")
        print(f"{model_name}_precision_macro: {model_metrics['precision_macro']:.4f}")
        print(f"{model_name}_recall_macro: {model_metrics['recall_macro']:.4f}")
        print(f"{model_name}_sustainability: {model_metrics['sustainability']:.4f}")
        print(f"{model_name}_runtime_mean_sec: {model_metrics['runtime_mean_sec']:.4f}")
        print(f"{model_name}_accuracy_std: {model_metrics['cv_accuracy_std']:.4f}")
        print(f"{model_name}_f1_macro_std: {model_metrics['cv_f1_macro_std']:.4f}")
        if "avg_thresholds" in model_metrics:
            print(f"{model_name}_avg_thresholds: {model_metrics['avg_thresholds']}")
    print(f"predictions_path: {pred_path}")
    print(f"summary_path: {summary_path}")
    print(f"plot_metrics_bar: {metric_plot_path}")
    print(f"plot_requested_metrics: {requested_metrics_plot_path}")
    print(f"plot_confusion_matrices: {confusion_plot_path}")
    print(f"plot_classwise_f1: {classwise_plot_path}")


if __name__ == "__main__":
    main()
