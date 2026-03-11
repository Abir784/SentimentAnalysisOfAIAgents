from __future__ import annotations

import html
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DetectorFactory.seed = 0

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
CONTINUE_RE = re.compile(r"\bContinue Reading\b.*", re.IGNORECASE | re.DOTALL)
MORE_FROM_RE = re.compile(r"\bMore from m/.*", re.IGNORECASE | re.DOTALL)
UI_GLYPH_RE = re.compile(r"[▲▼]")
MULTISPACE_RE = re.compile(r"\s+")
NON_WORD_SYMBOL_RE = re.compile(r"[^A-Za-z0-9\s\.,!?;:\-\'\"]+")
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.!?,;]")

NEGATION_CUES = {
    "aint",
    "barely",
    "cannot",
    "cant",
    "didnt",
    "doesnt",
    "dont",
    "hardly",
    "isnt",
    "lack",
    "lacked",
    "lacking",
    "lacks",
    "neither",
    "never",
    "no",
    "nobody",
    "none",
    "nor",
    "not",
    "nothing",
    "nowhere",
    "scarcely",
    "without",
    "wont",
    "wouldnt",
}

STOPWORD_EXCEPTIONS = NEGATION_CUES | {
    "against",
    "but",
    "down",
    "few",
    "further",
    "less",
    "least",
    "most",
    "more",
    "only",
    "over",
    "really",
    "so",
    "too",
    "under",
    "up",
    "very",
}

PUNCTUATION_TOKENS = {".", "!", "?", ",", ";"}
NEGATION_SCOPE_SIZE = 3
MIN_MODEL_TOKEN_COUNT = 3
MIN_MODEL_CHAR_COUNT = 20

_NLTK_READY = False
_STOPWORDS: set[str] | None = None
_LEMMATIZER = None
_POS_TAG = None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def basic_strip(text: str) -> str:
    cleaned = html.unescape(str(text or ""))
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = CONTINUE_RE.sub(" ", cleaned)
    cleaned = MORE_FROM_RE.sub(" ", cleaned)
    cleaned = UI_GLYPH_RE.sub(" ", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def preprocess_for_sentiment(text: str) -> str:
    cleaned = basic_strip(text)
    cleaned = NON_WORD_SYMBOL_RE.sub(" ", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def detect_language_safe(text: str) -> str:
    probe = basic_strip(text)[:1500]
    if len(probe.split()) < 3:
        return "unknown"
    try:
        return detect(probe)
    except LangDetectException:
        return "unknown"


def _ensure_nltk() -> None:
    global _NLTK_READY, _STOPWORDS, _LEMMATIZER, _POS_TAG
    if _NLTK_READY:
        return

    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

    _STOPWORDS = set(stopwords.words("english")) - STOPWORD_EXCEPTIONS
    _LEMMATIZER = WordNetLemmatizer()
    _POS_TAG = pos_tag
    _NLTK_READY = True


def _normalize_negation_forms(text: str) -> str:
    normalized = text.lower()
    replacements = {
        "can not": "cannot",
        "can't": "cannot",
        "cant": "cannot",
        "won't": "will not",
        "wont": "will not",
        "n't": " not",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"[^a-z\s\.!?,;']+", " ", normalized)
    normalized = MULTISPACE_RE.sub(" ", normalized)
    return normalized.strip()


def _wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return "a"
    if treebank_tag.startswith("V"):
        return "v"
    if treebank_tag.startswith("R"):
        return "r"
    return "n"


def _lemmatize_tokens(tokens: Sequence[str]) -> List[str]:
    if not tokens:
        return []

    _ensure_nltk()
    assert _LEMMATIZER is not None

    try:
        assert _POS_TAG is not None
        tagged = _POS_TAG(list(tokens))
        return [
            _LEMMATIZER.lemmatize(token, _wordnet_pos(tag))
            for token, tag in tagged
        ]
    except LookupError:
        return [_LEMMATIZER.lemmatize(token) for token in tokens]


def build_traditional_tokens(text: str) -> List[str]:
    normalized = _normalize_negation_forms(preprocess_for_sentiment(text))
    raw_tokens = TOKEN_RE.findall(normalized)
    lexical_tokens = [token for token in raw_tokens if token not in PUNCTUATION_TOKENS]
    lemmas = _lemmatize_tokens(lexical_tokens)

    _ensure_nltk()
    assert _STOPWORDS is not None

    lemma_iter = iter(lemmas)
    output_tokens: List[str] = []
    negation_window = 0

    for token in raw_tokens:
        if token in PUNCTUATION_TOKENS:
            negation_window = 0
            continue

        lemma = next(lemma_iter)
        if not lemma:
            continue

        if lemma in NEGATION_CUES:
            output_tokens.append("not")
            negation_window = NEGATION_SCOPE_SIZE
            continue

        if lemma in _STOPWORDS:
            continue

        if negation_window > 0:
            output_tokens.append(f"not_{lemma}")
            negation_window -= 1
        else:
            output_tokens.append(lemma)

    return output_tokens


def traditional_preprocess(text: str) -> str:
    return " ".join(build_traditional_tokens(text))


def label_from_compound(value: float) -> str:
    if value >= 0.05:
        return "positive"
    if value <= -0.05:
        return "negative"
    return "neutral"


def _score_texts(texts: Iterable[str]) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(str(text or "")) for text in texts]
    return pd.DataFrame(scores)


def build_polarity_dataframe(rows: Sequence[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df_raw = pd.DataFrame(rows)
    if df_raw.empty:
        empty_summary = {
            "raw_rows": 0,
            "non_english_removed": 0,
            "spam_marker_removed_after_language_filter": 0,
            "duplicate_removed_after_spam_filter": 0,
            "short_or_empty_removed_after_previous_filters": 0,
            "english_rows_kept": 0,
        }
        return df_raw, df_raw.copy(), empty_summary

    for column, default in (
        ("platform", "moltbook"),
        ("post_id", ""),
        ("thread_id", ""),
        ("comment_id", ""),
        ("parent_id", ""),
        ("level", 0),
        ("author_id", ""),
        ("relative_time", ""),
        ("is_verified", False),
        ("upvotes", None),
        ("text", ""),
        ("source_url", ""),
        ("fetched_at", ""),
        ("source_file", ""),
    ):
        if column not in df_raw.columns:
            df_raw[column] = default

    df_raw["text"] = df_raw["text"].fillna("").astype(str)
    df_raw["is_verified"] = df_raw["is_verified"].fillna(False).astype(bool)
    df_raw["upvotes_num"] = pd.to_numeric(df_raw["upvotes"], errors="coerce")
    df_raw["text_len_chars_raw"] = df_raw["text"].str.len()

    df_work = df_raw.copy()
    df_work["detected_language"] = df_work["text"].map(detect_language_safe)
    df_work["text_basic_clean"] = df_work["text"].map(preprocess_for_sentiment)
    df_work["text_traditional_clean"] = df_work["text"].map(traditional_preprocess)
    df_work["text_len_chars_basic_clean"] = df_work["text_basic_clean"].str.len()
    df_work["text_len_words_basic_clean"] = df_work["text_basic_clean"].str.split().str.len()
    df_work["text_len_chars_traditional_clean"] = df_work["text_traditional_clean"].str.len()
    df_work["text_len_words_traditional_clean"] = df_work["text_traditional_clean"].str.split().str.len()
    df_work["is_spam_marker"] = df_work["relative_time"].fillna("").astype(str).str.contains("spam", case=False)
    df_work["is_duplicate_comment"] = df_work.duplicated(subset=["platform", "post_id", "comment_id"])
    df_work["fails_min_length"] = (
        (df_work["text_len_words_traditional_clean"] < MIN_MODEL_TOKEN_COUNT)
        | (df_work["text_len_chars_traditional_clean"] < MIN_MODEL_CHAR_COUNT)
    )

    english_mask = df_work["detected_language"] == "en"
    step_1 = df_work[english_mask].copy()
    step_2 = step_1[~step_1["is_spam_marker"]].copy()
    step_3 = step_2[~step_2["is_duplicate_comment"]].copy()
    df_final = step_3[~step_3["fails_min_length"]].copy()

    raw_scores = _score_texts(df_final["text"])
    processed_scores = _score_texts(df_final["text_traditional_clean"])

    for column in ("neg", "neu", "pos", "compound"):
        df_final[f"raw_polarity_{column}"] = raw_scores[column].values
        df_final[f"processed_polarity_{column}"] = processed_scores[column].values

    df_final["raw_polarity_label"] = df_final["raw_polarity_compound"].map(label_from_compound)
    df_final["processed_polarity_label"] = df_final["processed_polarity_compound"].map(label_from_compound)
    df_final["polarity_compound_delta"] = df_final["processed_polarity_compound"] - df_final["raw_polarity_compound"]
    df_final["polarity_label_changed"] = df_final["raw_polarity_label"] != df_final["processed_polarity_label"]

    preprocess_summary = {
        "raw_rows": int(len(df_raw)),
        "non_english_removed": int((~english_mask).sum()),
        "spam_marker_removed_after_language_filter": int(step_1["is_spam_marker"].sum()),
        "duplicate_removed_after_spam_filter": int(step_2["is_duplicate_comment"].sum()),
        "short_or_empty_removed_after_previous_filters": int(step_3["fails_min_length"].sum()),
        "english_rows_kept": int(len(df_final)),
    }
    return df_raw, df_final, preprocess_summary


def _jsonl_output_paths(run_id: str, preprocessed_dir: Path, polarity_dir: Path) -> Dict[str, Path]:
    return {
        "preprocessed": preprocessed_dir / f"moltbook_comments_preprocessed_{run_id}.jsonl",
        "polarity": polarity_dir / f"moltbook_comments_polarity_{run_id}.jsonl",
        "summary": polarity_dir / f"moltbook_polarity_summary_{run_id}.json",
        "training_csv": preprocessed_dir / f"moltbook_training_ready_{run_id}.csv",
    }


def write_polarity_outputs(
    df_raw: pd.DataFrame,
    df_final: pd.DataFrame,
    preprocess_summary: Dict[str, Any],
    run_id: str,
    preprocessed_dir: Path,
    polarity_dir: Path,
) -> Dict[str, Any]:
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    polarity_dir.mkdir(parents=True, exist_ok=True)
    paths = _jsonl_output_paths(run_id, preprocessed_dir, polarity_dir)

    preprocessed_cols = [
        "platform",
        "post_id",
        "thread_id",
        "comment_id",
        "parent_id",
        "level",
        "author_id",
        "relative_time",
        "is_verified",
        "upvotes",
        "upvotes_num",
        "text",
        "text_basic_clean",
        "text_traditional_clean",
        "source_url",
        "fetched_at",
        "source_file",
        "detected_language",
        "is_spam_marker",
        "is_duplicate_comment",
        "text_len_chars_raw",
        "text_len_chars_basic_clean",
        "text_len_words_basic_clean",
        "text_len_chars_traditional_clean",
        "text_len_words_traditional_clean",
    ]

    polarity_cols = preprocessed_cols + [
        "raw_polarity_neg",
        "raw_polarity_neu",
        "raw_polarity_pos",
        "raw_polarity_compound",
        "raw_polarity_label",
        "processed_polarity_neg",
        "processed_polarity_neu",
        "processed_polarity_pos",
        "processed_polarity_compound",
        "processed_polarity_label",
        "polarity_compound_delta",
        "polarity_label_changed",
    ]

    training_cols = [
        "comment_id",
        "post_id",
        "thread_id",
        "parent_id",
        "level",
        "author_id",
        "is_verified",
        "upvotes_num",
        "detected_language",
        "text_basic_clean",
        "text_traditional_clean",
        "text_len_chars_traditional_clean",
        "text_len_words_traditional_clean",
        "raw_polarity_compound",
        "raw_polarity_label",
        "processed_polarity_compound",
        "processed_polarity_label",
        "polarity_compound_delta",
        "polarity_label_changed",
    ]

    df_final[preprocessed_cols].to_json(paths["preprocessed"], orient="records", lines=True, force_ascii=True)
    df_final[polarity_cols].to_json(paths["polarity"], orient="records", lines=True, force_ascii=True)
    df_final[training_cols].to_csv(paths["training_csv"], index=False, encoding="utf-8")

    raw_label_share = df_final["raw_polarity_label"].value_counts(normalize=True).to_dict()
    processed_label_share = df_final["processed_polarity_label"].value_counts(normalize=True).to_dict()
    summary_payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_row_count": int(len(df_raw)),
        "row_count_after_preprocessing": int(len(df_final)),
        "english_only": True,
        "scoring_comparison": {
            "raw_mean_compound": round(float(df_final["raw_polarity_compound"].mean()), 4) if not df_final.empty else 0.0,
            "processed_mean_compound": round(float(df_final["processed_polarity_compound"].mean()), 4) if not df_final.empty else 0.0,
            "mean_compound_delta": round(float(df_final["polarity_compound_delta"].mean()), 4) if not df_final.empty else 0.0,
            "label_change_rate": round(float(df_final["polarity_label_changed"].mean()), 4) if not df_final.empty else 0.0,
            "raw_label_share": {k: round(float(v), 4) for k, v in raw_label_share.items()},
            "processed_label_share": {k: round(float(v), 4) for k, v in processed_label_share.items()},
        },
        "preprocessing": {
            "language_filter": "detected_language == en",
            "remove_spam_marker_rows": True,
            "remove_duplicate_comments": True,
            "min_traditional_token_count": MIN_MODEL_TOKEN_COUNT,
            "min_traditional_char_count": MIN_MODEL_CHAR_COUNT,
            "negation_scope_size": NEGATION_SCOPE_SIZE,
            "stopword_policy": "NLTK English stopwords removed except negators and intensity/contrast terms needed for sentiment and modeling semantics.",
            "drop_counts": preprocess_summary,
            "cleaning_steps": [
                "html_unescape",
                "unicode_normalize_nfkc",
                "strip_markdown_links",
                "strip_urls",
                "strip_continue_reading_tail",
                "strip_more_from_tail",
                "strip_ui_vote_glyphs",
                "remove_non_word_symbols",
                "collapse_whitespace",
                "expand_negation_forms",
                "pos_aware_lemmatization",
                "negation_scope_prefixing",
                "policy_driven_stopword_removal",
            ],
        },
        "preprocessed_output": str(paths["preprocessed"]).replace("\\", "/"),
        "polarity_output": str(paths["polarity"]).replace("\\", "/"),
        "training_ready_csv": str(paths["training_csv"]).replace("\\", "/"),
    }
    paths["summary"].write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "paths": paths,
        "summary": summary_payload,
    }


def run_polarity_pipeline(
    input_path: Path,
    preprocessed_dir: Path,
    polarity_dir: Path,
    run_id: str | None = None,
) -> Dict[str, Any]:
    rows = read_jsonl(input_path)
    actual_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df_raw, df_final, preprocess_summary = build_polarity_dataframe(rows)
    outputs = write_polarity_outputs(
        df_raw=df_raw,
        df_final=df_final,
        preprocess_summary=preprocess_summary,
        run_id=actual_run_id,
        preprocessed_dir=preprocessed_dir,
        polarity_dir=polarity_dir,
    )
    return {
        "run_id": actual_run_id,
        "input_path": input_path,
        "raw_rows": len(df_raw),
        "rows_after_preprocessing": len(df_final),
        "paths": outputs["paths"],
        "summary": outputs["summary"],
    }