# Sentiment Dynamics in AI-to-AI Social Networks

Working title: `Sentiment Dynamics in AI-to-AI Social Networks: A Computational Analysis of MoltBook Conversations`

## Executive Summary (Short Abstract)
This study examines interaction patterns among AI agents on MoltBook, a public AI-native social platform in which autonomous accounts publish posts and exchange threaded comments. The analysis focuses on the content, polarity, and structural features of agent-to-agent discourse to characterize how conversational behavior varies across posts, threads, and authors. A reproducible natural language processing pipeline is used to collect, clean, preprocess, extract features, and apply rule-based sentiment tools, followed by transparent reporting of sentiment distributions and interaction patterns. The study is designed as a descriptive and exploratory computational investigation intended to build an empirical basis for understanding AI-agent social behavior in a multi-agent online environment.

Live dashboard: https://sentimentanalysisabir784.streamlit.app/

## Formal Research Questions
1. What are the dominant interaction patterns in AI-agent conversations on MoltBook?
  - Hypothesis: Agent interactions will exhibit non-random variation across posts and threads, with identifiable conversational clustering.
2. What is the sentiment distribution of AI-agent replies, and does it differ by post, thread, or author?
  - Hypothesis: Positive sentiment will be the most frequent class; neutral will be underrepresented.
3. Which observable conversation features are associated with positive, neutral, or negative replies?
  - Hypothesis: Longer and more context-dependent exchanges show greater sentiment variability than short or low-engagement replies.
4. Are the observed interaction patterns robust to preprocessing and rule-based method choices?
  - Hypothesis: Core descriptive patterns remain directionally stable across reasonable variants.

## How the Research Questions Will Be Answered

### RQ1 — Dominant Interaction Patterns Among AI Agents
We will construct a directed reply network in which nodes represent authors and edges represent observed reply relationships from parent comments to child comments. Edge weights will represent reply frequency between author pairs. Network structure will be summarized with core graph metrics, including in-degree, out-degree, reciprocity, and clustering coefficient, together with post-level and thread-level measures such as discussion depth, back-and-forth frequency, and concentration of replies around particular agents. Descriptive network and thread-level statistics will be used to identify recurring structural patterns. The outputs will include interaction summary tables and visualizations of the most common conversation structures.

### RQ2 — Sentiment Distribution and Group Variation
Using the VADER-derived labels described above, we will compute the overall sentiment distribution (positive, neutral, negative) for each comment. These will then be aggregated by post, thread, and author to compare sentiment proportions across groups. Confidence intervals and appropriate statistical comparisons will be reported to assess whether observed differences are statistically meaningful. The outputs will include class distribution charts and group-level comparison summaries.

### RQ3 — Observable Features Associated with Sentiment Classes
We will treat sentiment as the outcome variable and compare it descriptively against interpretable, observable features, including text length, thread depth, upvote count, and author verification status, among others. Feature-level distributions and subgroup contrasts will be summarized across sentiment classes using non-parametric descriptive statistics and visual comparisons. The outputs will include feature summary tables and class-specific interpretation notes.

### RQ4 — Robustness to Preprocessing and Rule-Based Choices
To validate the stability of our findings, we will rerun the analysis under alternative conditions, including raw versus cleaned text, stricter filtering thresholds, and multiple rule-based scoring views (VADER, SentiWordNet, and ensemble). The key question is whether the main conclusions remain directionally consistent across these variations. The output will be a robustness matrix clearly indicating which findings are stable and which are sensitive to methodological choices.

## Data Source and Data Summary
- Data source: public AI-to-AI conversations from MoltBook, collected in multiple crawl batches and consolidated into staged JSONL files.
- MoltBook context: MoltBook is an AI-native social platform where AI agents publish posts and interact through threaded comments, making it a suitable environment for studying machine-to-machine discourse patterns.
- Official website: https://www.moltbook.com/
- Unit of analysis: comment-level text, with post/thread context fields retained for aggregation.
- Current staged corpus: 1296 comments across 55 posts and 548 authors.
- Current preprocessed dataset: 1219 comments after language filtering and quality preprocessing for rule-based analysis.
- Label space: three-class sentiment (`negative`, `neutral`, `positive`).
- Core fields used: `comment_id`, `post_id`, `thread_id`, `author_id`, `text`, `upvotes`, `is_verified`, `fetched_at`.
- Data pipeline structure: raw collection -> staged consolidated comments -> preprocessed text -> EDA -> feature extraction -> rule-based sentiment outputs.

## Packages and Technologies Used
- Programming language: Python 3.x
- Data handling: pandas, numpy
- NLP preprocessing and sentiment: nltk, langdetect, vaderSentiment, SentiWordNet
- Rule-based sentiment tools: VADER, SentiWordNet, conservative ensemble
- Statistical testing: scipy.stats
- Graph analysis: networkx
- Visualization: matplotlib, seaborn
- File formats and storage: JSONL, CSV, JSON
- Config management: PyYAML
- Workflow environment: Jupyter notebooks + Python scripts (VS Code workspace)

## Methodology: Data Collection to Rule-Based Analysis
### 1. Data Collection and Staging
1. Collect raw MoltBook conversations into JSONL batches from public pages.
2. Consolidate raw batches into a staged comments file.
3. Preserve core metadata fields (comment_id, post_id, thread_id, author_id, text, upvotes, verification status, fetch timestamp).

### 2. Data Quality Control and Preprocessing
1. Remove malformed, empty, duplicate, and low-signal records.
2. Normalize text with lowercase conversion, punctuation/special-character cleanup, URL/hashtag/number/emoji removal, abbreviation expansion, tokenization, stopword policy, and lemmatization.
3. Store processed text and intermediate artifacts for reproducibility and audit.

Note: duplicate rows detected at staging are explicitly handled in preprocessing, and duplicate comments are removed before rule-based scoring.

### 3. Feature Extraction and Rule-Based Scoring
1. Extract interpretable comment-level features from cleaned text (character count, token count, unique-token ratio, punctuation intensity, uppercase ratio).
2. Score sentiment with three rule-based tools:
  - VADER
  - SentiWordNet
  - Ensemble decision rule
3. Compare method-level label shares and agreement rates.
4. Export feature tables, rule-based summaries, and diagnostic plots.


### 4. Evaluation and Reporting
1. Report key descriptive metrics: label shares by method, mean score by method, cross-method agreement, and subgroup sentiment contrasts.
2. Report RQ1 network metrics: node/edge counts, weighted interactions, reciprocity, clustering, and thread-level distributions.
3. Export summary JSON, CSV tables, and visual diagnostics for comparison and interpretation.

## Results (Rule-Based Pipeline, Run 20260419T092811Z)
Data basis: 1219 preprocessed English comments (rule-based run 20260419T092811Z) and interaction network run 20260419T092832Z.

### RQ1 — Dominant interaction patterns
Hypothesis verdict: SUPPORTED (exploratory, sequential-edge fallback).

Key metrics:
- Nodes = 548, edges = 1085, density = 0.00362.
- Reciprocity = 0.1493, global clustering coefficient = 0.0956.
- Largest weakly connected component: 544 nodes (99.27%) and 1083 edges (99.82%).
- Average shortest path length (LCC) = 4.4022, diameter = 13, degree assortativity = -0.0384.
- Community detection: 19 communities, modularity Q = 0.5741 (top-5 sizes: 69, 58, 55, 52, 47).
- KS test vs. Erdős–Rényi null on degree sequences: statistic = 0.4526, p = 6.18e-51.

Figures/tables:
- data/figures/rq1_degree_distribution_20260419T092832Z.png
- data/figures/rq1_thread_depth_histogram_20260419T092832Z.png
- data/figures/rq1_reply_concentration_20260419T092832Z.png
- data/figures/rq1_community_size_distribution_20260419T092832Z.png
- data/figures/rq1_network_sample_20260419T092832Z.png
- data/eda/moltbook_rq1_graph_metrics_20260419T092832Z.json
- data/eda/moltbook_rq1_post_thread_metrics_20260419T092832Z.csv

Interpretation:
AI-agent conversations are not random: the strong KS rejection and high modularity indicate structured clustering. At the same time, reciprocity (0.149) shows that fully bidirectional exchanges are relatively uncommon, so MoltBook discourse is better characterized as clustered but weakly reciprocal.

### RQ2 — Sentiment distribution and group variation
Hypothesis verdict: NOT SUPPORTED.

Key metrics:
- Ensemble sentiment shares: negative 0.0615, neutral 0.5480, positive 0.3905.
- Chi-square vs uniform baseline: chi2 = 450.6267, p = 1.4049e-98.
- VADER mean compound = 0.3386; SentiWordNet mean score = 0.0231.
- VADER/SentiWordNet agreement: major disagreement pattern is VADER=positive vs SWN=neutral (count = 305).
- Group-difference Kruskal-Wallis (positive proportion signal):
  - post_id: H = 110.8733, p = 2.81e-07
  - thread_id: H = 110.8733, p = 2.81e-07
  - author_id: H = 308.7110, p = 3.02e-10

HYPOTHESIS VERDICT BLOCK:
- Hypothesis: NOT SUPPORTED
- Neutral is the dominant class (54.8%), not positive (39.1%).
- Neutral is overrepresented relative to the uniform baseline, not underrepresented.
- Positive is the second most frequent class, not the first.
- This suggests AI-agent discourse on MoltBook is predominantly non-polar, with many informational/neutral exchanges.

Figures/tables:
- data/figures/rq2_corpus_distribution_20260419T092811Z.png
- data/figures/rq2_by_post_20260419T092811Z.png
- data/figures/rq2_by_author_entropy_20260419T092811Z.png
- data/figures/rq2_top_authors_20260419T092811Z.png
- data/figures/rq2_lexicon_agreement_heatmap_20260419T092811Z.png
- data/figures/rq2_group_post_stats_20260419T092811Z.csv
- data/figures/rq2_group_thread_stats_20260419T092811Z.csv
- data/figures/rq2_group_author_stats_20260419T092811Z.csv
- data/figures/rq2_group_kruskal_20260419T092811Z.csv

Interpretation:
The rule-based ensemble yields a neutral-dominant sentiment profile, and sentiment composition varies significantly by post/thread/author. Lexicon disagreement is non-trivial, reinforcing that method choice materially affects polarity assignment and should be treated as a robustness concern.

Gold-label status:
- Human-adjudicated labels are still unavailable in the current gold set.
- Annotation batch generated: data/gold/annotation_batch_01.csv with guide data/gold/annotation_batch_01_guide.txt.

### RQ3 — Observable features and sentiment association
Hypothesis verdict: PARTIALLY SUPPORTED.

Key metrics:
- Full feature-class test table: data/figures/rq3_feature_stats_20260419T092811Z.csv.
- Spearman (thread variability tests):
  - max_depth vs sentiment_std: r = -0.0096, p = 0.9488, 95% CI [-0.3637, 0.3227]
  - mean_len vs sentiment_std: r = 0.3935, p = 0.00621, 95% CI [0.1001, 0.6238]

Figures/tables:
- data/figures/rq3_feature_boxplots_20260419T092811Z.png
- data/figures/rq3_variability_scatter_20260419T092811Z.png
- data/figures/rq3_verified_vs_unverified_20260419T092811Z.png
- data/figures/rq3_feature_stats_20260419T092811Z.csv
- data/figures/rq3_variability_spearman_20260419T092811Z.csv

Interpretation:
Feature differences by sentiment class are detectable, but the variability hypothesis is only partly supported. Comment length is positively associated with sentiment variability across threads, while depth is not, suggesting context volume matters more than hierarchy depth under current metadata quality.

### RQ4 — Robustness to preprocessing and scoring choices
Hypothesis verdict: PARTIALLY SUPPORTED.

Key metrics:
- Stability matrix: data/figures/rq4_robustness_matrix_20260419T092811Z.csv.
- Coefficient of variation of positive share across variants: 0.2882.
- Max absolute delta of positive share from baseline: 0.3117.
- Sensitive variants: v3_vader_only, v4_swn_only.
- Baseline, basic_clean, and strict_filter variants remain neutral-dominant; scorer-only variants shift dominant class to positive.

Figures/tables:
- data/figures/rq4_robustness_heatmap_20260419T092811Z.png
- data/figures/rq4_robustness_matrix_20260419T092811Z.csv

Interpretation:
Core neutral dominance is stable under preprocessing changes but not under scorer substitution, so robustness holds for cleaning variants yet weakens when moving from ensemble to single-lexicon decisions. This identifies scorer choice as the primary sensitivity axis.

## Shortcomings in Current Results
1. Human-annotated gold labels are still missing for the current sample, so final Accuracy/Macro-F1/Weighted-F1/MCC and Cohen's Kappa against adjudicated labels remain pending.
2. RQ1 still depends on sequential edge fallback because direct parent-comment linkage is unresolved in staged metadata.
3. RQ3 hypothesis support is partial: variability tracks with mean length but not with thread depth under available depth signals.
4. Lexicon disagreement remains substantial and can shift downstream findings when single-method scorers are used.
5. Resource profiling is still partial: runtime is tracked, but memory and energy consumption are not yet integrated into comparative reporting.

## Archived: Phase 1 ML Experiments
These results are from a deprecated ML classification pipeline and do not reflect the current rule-based approach. Retained for record-keeping only.

| Model | Accuracy | Macro-F1 | Weighted-F1 | MCC | Status |
|---|---:|---:|---:|---:|---|
| Logistic Regression (archived) | 0.77 | N/A | N/A | N/A | Deprecated |
| SGD Classifier (archived) | N/A | 0.51 | N/A | N/A | Deprecated |

Notes:
- Confusion matrices and related ML metrics in earlier drafts belong to the archived ML phase only.
- Active pipeline for current claims is rule-based: VADER + SentiWordNet + ensemble.


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
- Comment-level: primary descriptive sentiment outcomes.
- Post-level: aggregated comment sentiment profiles.
- Optional: `agent_model`, moderation/report signals.
- Derived:
  - `sent_delta_raw_to_strict`

### 6A. Methodology (Phase 1)
2. Dual-path preprocessing:
  - Raw path: minimal normalization before scoring.
  - Strict path: lemmatization, negation-scope handling, and policy-based stopword removal.
3. Sentiment scoring:
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

## RQ-wise Graph Showcase (Answer-Oriented)

This section groups the most important visual outputs by research question so each RQ answer can be presented directly from figures.

### RQ1 — Dominant interaction patterns
Answer focus: interaction structure is clustered and non-random.

![RQ1 Network Topology Snapshot](data/figures/rq1_network_sample_20260419T092832Z.png)

![RQ1 Degree Distribution](data/figures/rq1_degree_distribution_20260419T092832Z.png)

![RQ1 Thread Depth Histogram](data/figures/rq1_thread_depth_histogram_20260419T092832Z.png)

![RQ1 Reply Concentration](data/figures/rq1_reply_concentration_20260419T092832Z.png)

![RQ1 Community Size Distribution](data/figures/rq1_community_size_distribution_20260419T092832Z.png)

### RQ2 — Sentiment distribution and group variation
Answer focus: neutral is dominant overall, with significant variation by post/thread/author.

![RQ2 Corpus Distribution](data/figures/rq2_corpus_distribution_20260419T092811Z.png)

![RQ2 Variation by Post](data/figures/rq2_by_post_20260419T092811Z.png)

![RQ2 Top Authors by Sentiment Profile](data/figures/rq2_top_authors_20260419T092811Z.png)

![RQ2 Author Entropy](data/figures/rq2_by_author_entropy_20260419T092811Z.png)

![RQ2 Lexicon Agreement Heatmap](data/figures/rq2_lexicon_agreement_heatmap_20260419T092811Z.png)

### RQ3 — Feature association with sentiment
Answer focus: feature effects are mixed; length-related variability is stronger than depth.

![RQ3 Feature Distributions by Sentiment Class](data/figures/rq3_feature_boxplots_20260419T092811Z.png)

![RQ3 Thread Variability Association](data/figures/rq3_variability_scatter_20260419T092811Z.png)

![RQ3 Verified vs Unverified Comparison](data/figures/rq3_verified_vs_unverified_20260419T092811Z.png)

### RQ4 — Robustness under methodological variants
Answer focus: findings are stable for preprocessing variants but sensitive to scorer choice.

![RQ4 Robustness Heatmap](data/figures/rq4_robustness_heatmap_20260419T092811Z.png)

Supporting matrix (table data): data/figures/rq4_robustness_matrix_20260419T092811Z.csv

### Optional single-slide summary mapping

- RQ1 supported: clustered interaction topology.
- RQ2 not supported: neutral dominates, not positive.
- RQ3 partially supported: variability tracks length more than depth.
- RQ4 partially supported: preprocessing robust, scorer-sensitive.








