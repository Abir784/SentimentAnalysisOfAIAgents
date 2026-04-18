# Sentiment Dynamics in AI-to-AI Social Networks

Working title: `Sentiment Dynamics in AI-to-AI Social Networks: A Computational Analysis of MoltBook Conversations`

## Executive Summary (Short Abstract)
This study examines interaction patterns among AI agents on MoltBook, a public AI-native social platform in which autonomous accounts publish posts and exchange threaded comments. The analysis focuses on the content, polarity, and structural features of agent-to-agent discourse to characterize how conversational behavior varies across posts, threads, and authors. A reproducible natural language processing pipeline is used to collect, clean, preprocess, and model the corpus, followed by transparent reporting of sentiment distributions, interaction patterns, and model performance. The study is designed as a descriptive and exploratory computational investigation intended to build an empirical basis for understanding AI-agent social behavior in a multi-agent online environment.

Live dashboard: https://sentimentanalysisabir784.streamlit.app/

## Research Questions
1. What are the dominant interaction patterns in AI-agent conversations on MoltBook?
  - Hypothesis: Agent interactions will exhibit non-random variation across posts and threads, with identifiable conversational clustering.
2. What is the sentiment distribution of AI-agent replies, and does it differ by post, thread, or author?
  - Hypothesis: Positive sentiment will be the most frequent class, while neutral sentiment will remain comparatively underrepresented.
3. Which observable conversation features are associated with positive, neutral, or negative replies?
  - Hypothesis: Longer and more context-dependent exchanges will show greater sentiment variability than short or low-engagement replies.
4. Are the observed interaction patterns robust to preprocessing and modeling choices?
  - Hypothesis: Core descriptive patterns will remain directionally stable across reasonable preprocessing variants, even if class-level performance changes.

## How the Research Questions Will Be Answered

### RQ1 — Dominant Interaction Patterns Among AI Agents
We will construct a directed reply network in which nodes represent authors and edges represent observed reply relationships from parent comments to child comments. Edge weights will represent reply frequency between author pairs. Network structure will be summarized with core graph metrics, including in-degree, out-degree, reciprocity, and clustering coefficient, together with post-level and thread-level measures such as discussion depth, back-and-forth frequency, and concentration of replies around particular agents. Descriptive network and thread-level statistics will be used to identify recurring structural patterns. The outputs will include interaction summary tables and visualizations of the most common conversation structures.

### RQ2 — Sentiment Distribution and Group Variation
Using the VADER-derived labels described above, we will compute the overall sentiment distribution (positive, neutral, negative) for each comment. These will then be aggregated by post, thread, and author to compare sentiment proportions across groups. Confidence intervals and appropriate statistical comparisons will be reported to assess whether observed differences are statistically meaningful. The outputs will include class distribution charts and group-level comparison summaries.

### RQ3 — Observable Features Associated with Sentiment Classes
We will treat sentiment as the outcome variable and model it against a set of interpretable, observable features, including text length, thread depth, upvote count, and author verification status, among others. Lightweight, interpretable models will be fitted, and feature effects will be compared across sentiment classes. The outputs will include feature importance or effect-size tables and class-specific interpretation summaries.

### RQ4 — Robustness to Preprocessing and Model Choices
To validate the stability of our findings, we will rerun the analysis under alternative conditions, including raw versus cleaned text, stricter filtering thresholds, and multiple model configurations. The key question is whether the main conclusions remain directionally consistent across these variations. The output will be a robustness matrix clearly indicating which findings are stable and which are sensitive to methodological choices.

## Data Source and Data Summary
- Data source: public AI-to-AI conversations from MoltBook, collected in multiple crawl batches and consolidated into staged JSONL files.
- MoltBook context: MoltBook is an AI-native social platform where AI agents publish posts and interact through threaded comments, making it a suitable environment for studying machine-to-machine discourse patterns.
- Official website: https://www.moltbook.com/
- Unit of analysis: comment-level text, with post/thread context fields retained for aggregation.
- Current staged corpus: 2163 comments across 55 posts and 548 authors.
- Current modeling dataset: 1040 labeled comments after preprocessing and quality filtering.
- Label space: three-class sentiment (`negative`, `neutral`, `positive`).
- Core fields used: `comment_id`, `post_id`, `thread_id`, `author_id`, `text`, `upvotes`, `is_verified`, `fetched_at`.
- Data pipeline structure: raw collection -> staged consolidated comments -> preprocessed text -> polarity and training-ready CSV artifacts.

## Packages and Technologies Used
- Programming language: Python 3.x
- Data handling: pandas, numpy
- NLP preprocessing and sentiment: nltk, langdetect, vaderSentiment
- ML modeling (lightweight): scikit-learn
  - TF-IDF features: `TfidfVectorizer`
  - Models: Logistic Regression, Linear SVM, SGDClassifier, Multinomial Naive Bayes
  - Custom model: Dual View Resonance (word-view + char-view + hybrid stack with neutral-guard rule)
  - Validation: StratifiedKFold, CalibratedClassifierCV
- Visualization: matplotlib, seaborn
- File formats and storage: JSONL, CSV, JSON
- Workflow environment: Jupyter notebooks + Python scripts (VS Code workspace)

## Methodology: Data Collection to Model Training
### 1. Data Collection and Staging
1. Collect raw MoltBook conversations into JSONL batches from public pages.
2. Consolidate raw batches into a staged comments file.
3. Preserve core metadata fields (comment_id, post_id, thread_id, author_id, text, upvotes, verification status, fetch timestamp).

### 2. Data Quality Control and Preprocessing
1. Remove malformed, empty, duplicate, and low-signal records.
2. Normalize text with lowercase conversion, punctuation/special-character cleanup, URL/hashtag/number/emoji removal, abbreviation expansion, tokenization, stopword policy, and lemmatization.
3. Store processed text and intermediate artifacts for reproducibility and audit.

Note: duplicate rows detected at staging are explicitly handled in preprocessing, and duplicate comments are removed before polarity scoring and model training.

### 3. Label Construction
Target labels are generated automatically from polarity scores using a fixed lexicon-threshold rule.

Labeling logic:
1. Compute polarity scores for each comment using VADER and extract the compound score $c \in [-1,1]$.
2. Apply a deterministic three-class mapping:
   - if $c \geq 0.05$, assign `positive`
   - if $c \leq -0.05$, assign `negative`
   - otherwise, assign `neutral`
3. Generate labels for both raw text and preprocessed text, then use `processed_polarity_label` as the default modeling target in the training-ready CSV.
4. Keep label space fixed to three classes: `negative`, `neutral`, `positive`.

Formal decision rule:

$$
y(c)=
\begin{cases}
  positive, & c \ge 0.05 \\
  negative, & c \le -0.05 \\
  neutral, & -0.05 < c < 0.05
\end{cases}
$$

Pseudocode:
```text
for each comment i:
  c_i = VADER_compound(text_traditional_clean_i)
  if c_i >= 0.05:
    label_i = positive
  elif c_i <= -0.05:
    label_i = negative
  else:
    label_i = neutral
```

### 4. Feature Engineering and Modeling
1. Transform text to TF-IDF features (unigram + bigram).
2. Train lightweight models suitable for local devices:
  - Logistic Regression (calibrated)
  - Linear SVM
  - SGD linear classifier
  - Multinomial Naive Bayes
3. Use stratified 5-fold cross-validation for robust estimates.
4. Apply minority-threshold tuning for probabilistic models to improve sensitivity on underrepresented classes.

### 4A. Dual View Resonance Model Specification
Proposed model: **Dual View Resonance (DVR)**

Core idea:
1. Learn sentiment from two synchronized text views of the same comment (`text_traditional_clean` and `text_basic_clean`) rather than a single lexical representation.
2. Use cross-view agreement/disagreement as an ambiguity signal and explicitly route low-confidence, high-ambiguity cases toward neutral-safe predictions.

Architecture:
1. View A encoder: TF-IDF word n-grams (1,2) over `text_traditional_clean` + SGD (`log_loss`, class-balanced).
2. View B encoder: TF-IDF character n-grams (3,5) over `text_basic_clean` + Logistic Regression (class-balanced).
3. Hybrid encoder: concatenated sparse matrix `[word_view || char_view]` + Multinomial Naive Bayes.
4. Meta-fusion layer: Logistic Regression over stacked probability outputs from the three base encoders plus auxiliary features:
  - confidence disagreement between word and char views,
  - `|polarity_compound_delta|`,
  - `text_len_words_traditional_clean`.
5. Neutral-guard rule: if meta confidence is below a fold-calibrated threshold and cross-view disagreement is small, prediction is mapped to `neutral`.

Algorithm (fold-level):
1. Split training data with stratified K-fold.
2. Fit word-view, char-view, and hybrid encoders on fold-train only.
3. Obtain train/test class probabilities from each base encoder.
4. Build meta-feature vectors by concatenating probabilities and auxiliary resonance features.
5. Fit a class-balanced meta Logistic Regression on fold-train meta-features.
6. Estimate neutral-guard threshold from fold-train confidence distribution.
7. Generate fold-test predictions via meta-model and apply neutral-guard post-rule.
8. Aggregate out-of-fold predictions across all folds and compute final metrics.

Pseudocode:
```text
Input:
  D = {(text_traditional_clean_i, text_basic_clean_i, y_i)} for i=1..N
  K = number of stratified folds

Initialize OOF predictions P_hat of size N

for each fold f in StratifiedKFold(D, K):
    Train, Test <- split(D, fold=f)

    # View encoders
    Xw_train <- TFIDF_word_ngrams(Train.text_traditional_clean)
    Xw_test  <- transform_word_ngrams(Test.text_traditional_clean)

    Xc_train <- TFIDF_char_ngrams(Train.text_basic_clean)
    Xc_test  <- transform_char_ngrams(Test.text_basic_clean)

    Xh_train <- concat_sparse(Xw_train, Xc_train)
    Xh_test  <- concat_sparse(Xw_test, Xc_test)

    M_word   <- fit SGD(log_loss, class_balanced) on (Xw_train, y_train)
    M_char   <- fit LogisticRegression(class_balanced) on (Xc_train, y_train)
    M_hybrid <- fit MultinomialNB on (Xh_train, y_train)

    Pw_train, Pw_test <- predict_proba(M_word, Xw_train, Xw_test)
    Pc_train, Pc_test <- predict_proba(M_char, Xc_train, Xc_test)
    Ph_train, Ph_test <- predict_proba(M_hybrid, Xh_train, Xh_test)

    R_train <- |max(Pw_train) - max(Pc_train)|
    R_test  <- |max(Pw_test) - max(Pc_test)|

    Z_train <- concat(Pw_train, Pc_train, Ph_train, R_train,
                      abs(polarity_compound_delta_train),
                      text_len_words_traditional_clean_train)
    Z_test  <- concat(Pw_test, Pc_test, Ph_test, R_test,
                      abs(polarity_compound_delta_test),
                      text_len_words_traditional_clean_test)

    M_meta <- fit LogisticRegression(class_balanced) on (Z_train, y_train)
    Q_test <- predict_proba(M_meta, Z_test)

    tau_neutral <- calibrate_threshold(max_confidence_train(M_meta, Z_train), y_train)

    y_pred <- argmax_labels(Q_test)
    for each sample j in Test:
        if max(Q_test[j]) < tau_neutral and R_test[j] < resonance_cutoff:
            y_pred[j] <- neutral

    write y_pred into OOF slots of fold f

Return OOF predictions P_hat and evaluation metrics
```

### 5. Evaluation and Reporting
1. Report five key metrics: Accuracy, F1 Score (macro), Precision (macro), Recall (macro), Sustainability.
2. Define Sustainability as a runtime-efficiency indicator normalized to [0, 1], where higher means faster and more device-friendly.
3. Export prediction tables, summary JSON, and visual diagnostics for comparison and interpretation.

## Results (Latest Deep-Enabled Run)
Data: 1040 labeled comments, 5-fold stratified cross-validation for lightweight models, plus full-dataset pretrained inference for deep models.

Run ID: `20260329T101730Z` (latest run with deep models enabled)

1. Best Accuracy: Logistic Regression (0.7750)
2. Best Macro F1: SGD Linear (0.5158)
3. Best Sustainability: Multinomial Naive Bayes (1.0000)
4. Custom model (Dual View Resonance) reached macro F1 = 0.5137 with explicit neutral-guard behavior, accuracy = 0.7452, and moderate runtime cost.
5. Deep transformer baselines still underperformed on this dataset under current label mapping (accuracy: 0.1913 and 0.1510), indicating domain adaptation and label calibration are still needed before deployment.

### Model Results Table

| Model | Accuracy | F1 Score (Macro) | Precision (Macro) | Recall (Macro) | Sustainability |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (calibrated) | 0.7750 | 0.4937 | 0.8268 | 0.4644 | 0.9993 |
| Linear SVM | 0.7654 | 0.5060 | 0.6898 | 0.4775 | 1.0000 |
| SGD Linear | 0.7577 | 0.5158 | 0.5925 | 0.5008 | 1.0000 |
| Multinomial Naive Bayes | 0.7365 | 0.3263 | 0.5444 | 0.3565 | 1.0000 |
| Dual View Resonance (custom) | 0.7452 | 0.5137 | 0.5323 | 0.5134 | 0.9971 |
| Deep: CardiffNLP Twitter-RoBERTa | 0.1913 | 0.2327 | 0.5085 | 0.4535 | 0.0000 |
| Deep: BERTweet Sentiment | 0.1510 | 0.1798 | 0.5415 | 0.4021 | 0.2601 |

### RQ Results

**RQ1 (Dominant interaction patterns among AI agents):** The interaction-network results are directionally consistent with the RQ1 hypothesis that interaction structure is non-random and cluster-like across threads. In the latest run, the graph contains 548 author nodes and 1121 directed edges (weighted interactions = 2044), with reciprocity = 0.1552 and average clustering coefficient = 0.1075, indicating measurable repeated interaction loops and local clustering rather than uniform random exchange. Because explicit parent-child reply links are currently sparse in raw staging, the present graph was constructed in sequential thread fallback mode; therefore, this should be interpreted as strong exploratory support for RQ1, pending stronger direct reply-edge coverage in future data collection.

**RQ1 Hypothesis Decision:** **Provisionally accepted (exploratory)**. The observed interaction structure supports the hypothesis direction (non-random variation with clustering), but final confirmation remains conditional on improved direct parent-child reply linkage in future data runs.

### Relevant Graphs
RQ1 interaction network topology (latest run):

![RQ1 interaction network topology](data/eda/moltbook_interaction_network_topology_20260418T164728Z.png)

RQ1 interaction metric distributions (latest run):

![RQ1 interaction metric distributions](data/eda/moltbook_interaction_network_distributions_20260418T164728Z.png)

Requested metrics dashboard (Accuracy, F1, Precision, Recall, Sustainability):

![Requested metrics dashboard](data/eda/moltbook_model_requested_metrics_20260329T101730Z.png)

Confusion matrices across models:

![Confusion matrices](data/eda/moltbook_model_confusion_matrices_20260329T101730Z.png)

Class-wise F1 comparison:

![Class-wise F1](data/eda/moltbook_model_classwise_f1_20260329T101730Z.png)

Latest summary artifact: `data/modeling/moltbook_model_summary_20260329T101730Z.json`
Latest predictions artifact: `data/modeling/moltbook_model_predictions_20260329T101730Z.csv`

## Shortcomings in Current Results
1. Neutral class performance is still weak because class support is low relative to positive samples (neutral support remains very limited).
2. Accuracy is acceptable, but macro-level metrics still show imbalance sensitivity and limited minority recall.
3. Latest crawl expansion increased staged duplicates (552 duplicate rows detected before strict preprocessing), requiring stronger dedup controls earlier in the pipeline.
4. The custom dual-view model improves interpretability of ambiguity handling (neutral-guard), but still trades off runtime efficiency.
5. Sustainability is runtime-based only; it does not yet include memory footprint and energy measurements.


<!-- ## Immediate Improvement Plan
1. Increase minority-class coverage via targeted data collection and/or controlled resampling.
2. Add memory-usage logging to extend Sustainability beyond runtime.
3. Build a small manually reviewed validation subset to audit neutral-label quality and error patterns.
4. Keep lightweight models as deployment baseline and use larger models only for periodic robustness checks. -->
<!-- 
## Phase 1 Research Design Matrix

### Study Scope
Primary objective: characterize sentiment structure in AI-agent discourse on MoltBook using a transparent, reproducible NLP pipeline.

Phase 1 focus: descriptive and methodological analysis only.

Out of Phase 1 scope: causal contagion claims, full thread-dynamics inference, and safety early-warning deployment.

### Matrix

| Module | Research Question | Testable Hypotheses | Unit of Analysis | Required Data Fields | Operationalization | Methods | Evaluation / Outputs | Key Validity Risks | Mitigations |
|---|---|---|---|---|---|---|---|---|---|
| M0: Data and Sampling | Can we build a reproducible MoltBook corpus for AI-agent discourse? | H0.1: Current crawl captures stable descriptive estimates under re-sampling. | Post, thread | `post_id`, `thread_id`, `comment_id`, `author_id`, `text`, `upvotes`, `is_verified`, `fetched_at` | Repeated-batch collection and deduped consolidated staging | Data audit, missingness checks, duplicate diagnostics, sensitivity re-sampling | Reproducible dataset card and limitations report | Scrape bias, missing metadata | Multi-batch collection, explicit missingness reporting |
| M1: Descriptive Sentiment Atlas | What is overall sentiment distribution in MoltBook? | H1.1: Positive sentiment is the modal class. H1.2: Distribution is robust to stricter preprocessing choices. | Comment and post | M0 + polarity scores | Polarity (`neg/neu/pos`) + compound intensity; post-level aggregation | Lexicon pipeline (VADER) on raw text and strict preprocessed text; bootstrap CIs | Distribution plots, post-level summaries, robustness deltas | Domain shift in sentiment models | Manual spot-checking and model-choice transparency |
| M2: Exploratory Structure (Non-causal) | Which interaction signatures appear in available metadata? | H2.1: Verified and non-verified agent comments differ in score distribution. H2.2: High-volume posts show heterogeneous sentiment patterns. | Comment and post | M0 + polarity + `is_verified` + `upvotes` | Grouped descriptive contrasts only (no causal interpretation) | Non-parametric tests, effect sizes, visualization | Exploratory appendix tables and plots | Confounding and missing thread structure | Clearly label exploratory and non-causal scope |

### Measurement Plan

| Construct | Metric | Notes |
|---|---|---|
| Sentiment polarity | `P(pos), P(neu), P(neg)` | Report with 95% CI |
| Sentiment intensity | Continuous score in [-1, 1] or [0, 1] | Keep model-specific scale mapping |
| Polarization (exploratory) | Variance / entropy complement | Post-level aggregates |
| Robustness shift | `processed_compound - raw_compound` | Preprocessing sensitivity metric |

### Data Requirements Checklist

| Priority | Field | Required For |
|---|---|---|
| Critical | `text`, `post_id`, `thread_id`, `comment_id`, `author_id` | Core Phase 1 analyses |
| High | `upvotes`, `is_verified`, `fetched_at` | Stratified descriptive analyses |
| Future-critical | `timestamp`, `reply_to`, `topic` | Phase 2 dynamics and mechanism tests |
| Optional but high-value | `agent_model` / agent type metadata | Model-stratified analysis |
| Optional | Moderation/report labels | Future safety validation |

### Identification and Inference Strategy

| Question Type | Preferred Inference |
|---|---|
| Descriptive prevalence | Bootstrap confidence intervals on corpus and post-level estimates |
| Group differences (within MoltBook) | Stratified comparisons + robust effect sizes |
| Predictors of shifts | Interpretable supervised models + SHAP |
| Mechanism (contagion) | Deferred to Phase 2 pending reply/timestamp coverage |
| Safety monitoring | Deferred to Phase 2 pending topic and toxicity signals |

### Quality Control and Reproducibility

| Component | Requirement |
|---|---|
| Annotation | Gold set with inter-annotator agreement (`Cohen's kappa`) |
| Preprocessing | Public, versioned pipeline and deterministic tokenization rules |
| Validation | Cross-model sentiment robustness and calibration report |
| Statistics | Multiple-comparison control (`Benjamini-Hochberg`) |
| Transparency | Preregistered hypotheses and decision thresholds |
| Ethics | Privacy-preserving storage, platform ToS compliance, no deanonymization |

### Deliverables by Paper Section

| Paper Section | Deliverable |
|---|---|
| Data | Dataset card + collection protocol |
| Methods | Sentiment pipeline and robustness methodology |
| Results 1 | Sentiment distribution atlas within MoltBook |
| Results 2 | Raw-vs-preprocessed robustness analysis |
| Appendix | Exploratory subgroup analyses and limitations |

### Minimal Feasible Version
1. M0 + M1 with strict reproducibility controls.
2. Include M2 exploratory subgroup analyses as non-causal appendix.
3. Defer dynamics, contagion, and safety forecasting to Phase 2.

### Current Data Limits and Identifiable Claims
- Identifiable now:
  - Corpus-level sentiment prevalence and compound-score distribution.
  - Post-level sentiment aggregation and subgroup contrasts (`is_verified`, engagement, high-volume posts).
  - Sensitivity of conclusions to preprocessing policy (raw vs strict traditional NLP pipeline).
- Not identifiable now:
  - Reply-edge contagion effects.
  - Turn-level escalation/convergence dynamics.
  - Topic-week safety stress trajectories.
- Main blockers:
  - Missing reliable `reply_to` structure.
  - Missing canonical `timestamp` and topic taxonomy for each comment.
  - No toxicity/moderation signal integration.

### Phase 2 Roadmap
1. Data schema upgrade:
  - Add canonical timestamps, topic labels, and reply-edge extraction.
  - Preserve backwards compatibility with current staged schema.
2. Annotation and validation:
  - Build a stratified gold set for sentiment (and optional toxicity).
  - Report inter-annotator agreement and calibration curves.
3. Dynamics and mechanism modeling:
  - Estimate escalation hazard and lagged reply influence using fixed-effects models.
  - Add placebo lag tests and shuffled exposure tests.
4. Safety layer:
  - Construct topic-week Safety Stress Index with changepoint detection.
  - Validate thresholds through false-alarm analysis.

---

## Preregistration Template (Paper-Ready)
Project: `Sentiment Dynamics in AI-to-AI Social Networks (MoltBook)`

### 1. Study Overview
- Objective: Quantify sentiment structure in AI-to-AI discourse on MoltBook with transparent preprocessing and robustness checks.
- Design: Observational computational social science study with descriptive and methodological components.
- Primary platform: MoltBook (AI-only social network).

### 2. Research Questions
1. What is the sentiment distribution in MoltBook overall and across available metadata strata?
2. How sensitive are sentiment outcomes to stricter NLP preprocessing choices?
3. Which descriptive subgroup differences are observable with current metadata?

### 3. Hypotheses
- H1: Positive sentiment is the modal class in MoltBook comments after quality filtering.
- H2: Core sentiment distribution findings remain directionally stable under strict preprocessing.
- H3: Verified-status and engagement strata exhibit measurable descriptive differences in sentiment.

### 4. Data Sources and Inclusion Rules
- Inclusion:
  - Public posts only.
  - Language: English (or explicitly multilingual with language-specific models).
  - Time window: predefined fixed interval (for example, 12 months).
- Exclusion:
  - Deleted/inaccessible content.
  - Duplicates/near-duplicates beyond threshold.
  - Non-text or empty-text posts.

### 5. Units of Analysis
- Comment-level: primary descriptive sentiment outcomes.
- Post-level: aggregated comment sentiment profiles.

### 6. Variables
- Core IDs: `post_id`, `thread_id`, `comment_id`, `agent_id/user_id`.
- Context: `text`, `upvotes`, `is_verified`, `fetched_at`.
- Optional: `agent_model`, moderation/report signals.
- Derived:
  - `sent_polarity` (`neg/neu/pos`)
  - `sent_intensity` (continuous)
  - `preprocessing_variant` (`raw`, `strict`)
  - `sent_delta_raw_to_strict`

### 6A. Methodology (Phase 1)
1. Data ingestion and quality control:
  - Read consolidated staged comments.
  - Remove exact duplicate comments and malformed/empty rows.
  - Restrict to English via deterministic language filter.
2. Dual-path preprocessing:
  - Raw path: minimal normalization before scoring.
  - Strict path: lemmatization, negation-scope handling, and policy-based stopword removal.
3. Sentiment scoring:
  - Score both paths with the same VADER model to isolate preprocessing effects.
  - Produce polarity labels and compound scores for each comment.
4. Statistical analysis:
  - Estimate corpus-level label shares and compound-score summaries with bootstrap CIs.
  - Compare subgroup distributions (`is_verified`, engagement bins, top-volume posts).
  - Report effect sizes and practical differences before significance tests.
5. Reproducibility controls:
  - Fixed random seeds and versioned scripts.
  - Export machine-readable artifacts (`jsonl`, `csv`, summary `json`) for auditability.

### 7. Sentiment Measurement Plan
- Primary model: lexicon/rule-based baseline (VADER) with strict and raw preprocessing variants.
- Secondary model: transformer sentiment classifier for Phase 2 robustness extension.
- Calibration:
  - Gold labeled subset (stratified by post volume, verification status, and time windows where available).
  - Report macro-F1, calibration error, confusion matrix.
- Primary outcome metric:
  - Polarity distribution across comments and post-level aggregates within MoltBook.
- Secondary metrics:
  - Intensity mean/variance, entropy, polarization index, volatility.

### 8. Annotation Protocol (Gold Set)
- Sample size: predefine (for example, 3,000 to 10,000 posts depending resources).
- Label schema: `neg`, `neu`, `pos` (+ optional confidence, sarcasm flag).
- Annotators: minimum 2 independent + adjudication.
- Agreement threshold:
  - Target `Cohen's kappa >= 0.70`.
  - If below, revise guidelines and relabel pilot subset.

### 9. Primary Analyses
1. Descriptive atlas:
  - Label shares and compound distributions with confidence intervals.
  - Post-level aggregated sentiment summaries.
2. Robustness analysis:
  - Raw vs strict preprocessing deltas in scores and labels.
3. Exploratory subgroup analysis:
  - Descriptive contrasts by verification status and engagement strata.

### 10. Event Definitions (Preregistered)
- Phase 1 does not preregister causal or turn-level events due to current schema limits.
- Event definitions for escalation, convergence, and contagion are deferred to Phase 2 after reply-edge and timestamp upgrades.

### 11. Statistical Plan
- Confidence intervals: bootstrap or robust sandwich standard errors.
- Multiple testing control: Benjamini-Hochberg FDR.
- Reporting: effect sizes first, p-values second.
- Model diagnostics:
  - Collinearity checks.
  - Residual and calibration diagnostics.
  - Out-of-time validation split.

### 12. Robustness and Ablations
- Alternate sentiment models and thresholds.
- Topic-removal sensitivity (drop largest topics).
- Preprocessing-policy sensitivity (raw vs strict path).
- Engagement-controlled subsets.
- Temporal robustness (early vs late period splits).
- Contagion placebo tests deferred to Phase 2.

### 13. Bias, Ethics, and Safety
- Respect platform terms and legal constraints.
- No deanonymization attempts.
- Store only required fields; redact personally identifying text if encountered.
- Document model bias risks (dialect, sarcasm, topic framing).
- Release only aggregate statistics where needed.

### 14. Reproducibility Commitments
- Versioned pipeline (data processing, modeling, analysis scripts).
- Deterministic seeds and environment lockfile.
- Dataset card with missingness and collection limitations.
- Public code (where permissible) + synthetic examples if raw text cannot be shared.

### 15. Threats to Validity (Declared Up Front)
- Construct validity: sentiment models may misread AI style, irony, role-play.
- Internal validity: homophily and topic drift can mimic contagion.
- External validity: MoltBook may not represent all AI-agent ecosystems.
- Platform-specific affordances may shape discourse in ways that limit generalization beyond MoltBook.

### 16. Minimum Success Criteria
- Reliable sentiment measurement on validation set (predefined performance floor).
- Stable descriptive conclusions across preprocessing variants.
- Transparent limitations and failure modes documented.

### 17. Planned Outputs
- Main paper tables:
  - Sentiment distribution by topic within MoltBook.
  - Raw-vs-strict preprocessing robustness deltas.
  - Exploratory subgroup contrasts (verification/engagement).
- Figures:
  - Corpus-level polarity distributions.
  - Raw-vs-strict comparison plots.
- Appendix:
  - Annotation guidelines, ablations, diagnostics, and Phase 2 roadmap. -->
