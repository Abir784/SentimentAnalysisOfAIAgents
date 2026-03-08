# Sentiment Dynamics in AI-to-AI Social Networks

Working title: `Sentiment Dynamics in AI-to-AI Social Networks: A Computational Analysis of MoltBook Conversations`

## Full Research Design Matrix

### Study Scope
Primary objective: characterize sentiment in AI-agent discourse, compare it with human discourse, and test one dynamic mechanism (contagion) plus safety implications.

### Matrix

| Module | Research Question | Testable Hypotheses | Unit of Analysis | Required Data Fields | Operationalization | Methods | Evaluation / Outputs | Key Validity Risks | Mitigations |
|---|---|---|---|---|---|---|---|---|---|
| M0: Data and Sampling | Can we build a representative MoltBook corpus and matched human baseline corpus? | H0.1: Sampling frame reproduces platform-level topic/time distribution. | Post, thread, topic-week | `post_id`, `thread_id`, `agent_id`, `timestamp`, `topic`, `text`, `reply_to`, `upvotes`, `agent_model` (if available), plus matched Reddit/X fields | Stratified sampling by topic and week; matched topic taxonomy across platforms | Data audit, distribution checks, KL divergence and chi-square for sample vs population | Reproducible dataset card; representativeness report | API/scrape bias, missing metadata, bot filtering differences | Multi-source collection, missingness diagnostics, sensitivity subsets |
| M1: Descriptive Sentiment Atlas | What is overall sentiment distribution and how does it vary by topic/time? | H1.1: MoltBook has higher neutral share than human baseline. H1.2: Topic-level sentiment variance is lower in MoltBook. | Post and topic-week | M0 + sentiment labels/scores | Polarity (`neg/neu/pos`) + intensity score; topic-week aggregates | Ensemble sentiment pipeline (lexicon + transformer + calibration); mixed-effects regression by topic/time | Distribution plots, topic heatmaps, temporal trend estimates | Domain shift in sentiment models | Manual validation set, calibration curves, robustness across models |
| M2: Thread Dynamics | How does sentiment evolve across multi-turn AI threads? | H2.1: AI threads converge to neutral faster than human threads. H2.2: Escalation probability increases with thread length and participant count. | Thread-turn sequence | M0 + turn index + sentiment per turn | Trajectory classes: `stable`, `escalating`, `de-escalating`, `oscillating` | Sequence clustering, Markov transition models, survival models for escalation events | Transition matrices, hazard ratios, trajectory typology | Confounding from topic mix and thread truncation | Topic-stratified models, censoring corrections, fixed effects |
| M3: Comparative Benchmark (AI vs Human) | How do AI-agent sentiment patterns differ from human discourse on comparable topics? | H3.1: MoltBook has lower negative tail risk. H3.2: Sentiment persistence (autocorrelation) is higher in MoltBook. | Topic-week, thread | Matched corpora from M0, same preprocessing | Matched topic-time bins; length-controlled subsets | Propensity score matching / exact matching, distributional tests, effect sizes, permutation tests | Cross-platform effect size table with CIs | Non-equivalent platform affordances (votes, moderation) | Include platform controls, run placebo topics, report residual confounding bounds |
| M4: Contagion Mechanism | Does sentiment in one agent message influence subsequent replies? | H4.1: Exposure to negative sentiment increases probability of negative next-turn response. H4.2: Influence decays with lag. | Reply edge and lagged turn | `reply_to`, timestamps, sentiment lag features, agent IDs | Exposure variables from parent and recent thread context; outcome = next-turn sentiment | Temporal logistic models, Hawkes process, Granger-style lag tests, randomization inference | Causal-style influence estimates by lag and topic | Homophily vs contagion, simultaneity | Matched exposure windows, agent fixed effects, placebo lag tests |
| M5: Safety and Risk Layer | Can sentiment dynamics reveal emergent safety concerns? | H5.1: Certain topics show persistent high-polarization/high-negativity risk states. | Topic-week, thread | Sentiment + toxicity/abuse classifiers + upvotes/report signals if available | Composite `Safety Stress Index` = weighted z-scores (negativity, volatility, polarization, toxicity co-signal) | Risk scoring, changepoint detection, early-warning model | Topic risk dashboard, top-risk episodes, false-alarm analysis | Classifier bias and threshold sensitivity | Threshold sweep, human spot-check, subgroup fairness checks |

### Measurement Plan

| Construct | Metric | Notes |
|---|---|---|
| Sentiment polarity | `P(pos), P(neu), P(neg)` | Report with 95% CI |
| Sentiment intensity | Continuous score in [-1, 1] or [0, 1] | Keep model-specific scale mapping |
| Polarization | Bimodality index / variance / entropy complement | Topic-week level |
| Volatility | Standard deviation of sentiment in rolling windows | Thread and topic-week |
| Persistence | Lag-1 autocorrelation | Thread level |
| Escalation | `Pr(neg_t+1 | context_t)` | Event definition preregistered |
| Contagion strength | Marginal effect of parent sentiment on reply sentiment | With fixed effects |
| Safety stress | Composite index percentile | Publish full formula |

### Data Requirements Checklist

| Priority | Field | Required For |
|---|---|---|
| Critical | `text`, `timestamp`, `thread_id`, `reply_to`, `topic` | All modules |
| Critical | `post_id`, `agent_id` | Dynamics + contagion |
| High | `upvotes` / engagement | Predictive controls, safety interpretation |
| High | Human baseline corpus with aligned topics/time | Comparative module |
| Optional but high-value | `agent_model` / agent type metadata | Model-stratified analysis |
| Optional | Moderation/report labels | Safety validation |

### Identification and Inference Strategy

| Question Type | Preferred Inference |
|---|---|
| Descriptive prevalence | Weighted estimates by topic-time strata |
| Group differences (AI vs human) | Matched comparisons + robust effect sizes |
| Predictors of shifts | Interpretable supervised models + SHAP |
| Mechanism (contagion) | Lagged models with agent/thread fixed effects + placebo tests |
| Safety monitoring | Out-of-sample early-warning performance |

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
| Methods | Sentiment pipeline and causal/temporal modeling spec |
| Results 1 | Sentiment atlas across topics and time |
| Results 2 | AI-vs-human comparative effect sizes |
| Results 3 | Thread trajectory taxonomy and contagion estimates |
| Applied | Safety Stress Index and risk episodes |
| Appendix | Robustness, ablations, error analysis, limitations |

### Minimal Feasible Version
1. M0 + M1 + M3 only (descriptive + comparative core).
2. Add one dynamic test from M2 (`escalation hazard`) instead of full M4.
3. Keep M5 as exploratory appendix.

---

## Preregistration Template (Paper-Ready)
Project: `Sentiment Dynamics in AI-to-AI Social Networks (MoltBook)`

### 1. Study Overview
- Objective: Quantify sentiment structure and dynamics in AI-to-AI discourse on MoltBook, compare with matched human discourse, and test whether sentiment contagion-like effects exist.
- Design: Observational computational social science study with descriptive, comparative, and mechanism-testing components.
- Primary platform: MoltBook (AI-only social network).
- Comparator platforms: Reddit and/or X (topic- and time-matched subsets).

### 2. Research Questions
1. What is the sentiment distribution in MoltBook overall and by topic/time?
2. How does sentiment evolve within multi-turn AI-agent threads?
3. How does MoltBook sentiment differ from matched human discourse?
4. Does prior sentiment exposure in-thread predict subsequent reply sentiment (contagion test)?
5. Which topics/time windows show elevated sentiment-related safety risk?

### 3. Hypotheses
- H1: MoltBook has a higher neutral-sentiment proportion than matched human discourse.
- H2: MoltBook threads show lower escalation rate (negative drift) than matched human threads.
- H3: Sentiment autocorrelation within threads is higher in MoltBook than human baselines.
- H4: Parent-turn negative sentiment increases probability of negative reply sentiment (contagion-consistent effect), with decay across lag.
- H5: A subset of topics exhibits persistent high sentiment volatility/polarization, indicating elevated safety stress.

### 4. Data Sources and Inclusion Rules
- Inclusion:
  - Public posts only.
  - Language: English (or explicitly multilingual with language-specific models).
  - Threads with at least 3 turns for dynamics analyses.
  - Time window: predefined fixed interval (for example, 12 months).
- Exclusion:
  - Deleted/inaccessible content.
  - Duplicates/near-duplicates beyond threshold.
  - Non-text or empty-text posts.
- Matching constraints for human baseline:
  - Topic mapping to MoltBook taxonomy.
  - Similar post-length bins.
  - Similar calendar periods.
  - Optional engagement matching (upvote/reply strata).

### 5. Units of Analysis
- Post-level: sentiment prevalence and intensity.
- Thread-turn level: trajectory and lag effects.
- Topic-week level: temporal trends and safety index.
- Reply-edge level: parent-to-child sentiment influence.

### 6. Variables
- Core IDs: `post_id`, `thread_id`, `reply_to`, `agent_id/user_id`.
- Context: `timestamp`, `topic`, `text`, `upvotes`, `reply_count`.
- Optional: `agent_model`, moderation/report signals.
- Derived:
  - `sent_polarity` (`neg/neu/pos`)
  - `sent_intensity` (continuous)
  - `turn_index`
  - `lagged_sentiment` (parent and k-lag context)
  - `thread_participant_count`, `thread_length`

### 7. Sentiment Measurement Plan
- Primary model: transformer sentiment classifier fine-tuned/calibrated for short social text.
- Secondary model: lexicon/rule-based baseline.
- Calibration:
  - Gold labeled subset (stratified by topic/platform/time).
  - Report macro-F1, calibration error, confusion matrix.
- Primary outcome metric:
  - Polarity distribution by platform/topic/time.
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
   - Weighted prevalence estimates by topic-week.
   - Mixed-effects regression (`sentiment ~ topic + time + (1|thread)` as appropriate).
2. Comparative analysis:
   - Matched sample tests (propensity/exact matching).
   - Effect sizes with 95% CI and permutation robustness.
3. Thread dynamics:
   - Transition matrices; hazard/survival model for escalation event.
4. Contagion test:
   - Lagged logistic/ordinal models with thread and agent fixed effects.
   - Hawkes-style sensitivity analysis.
5. Safety layer:
   - Composite Safety Stress Index from negativity, volatility, polarization, and toxicity co-signal.

### 10. Event Definitions (Preregistered)
- Escalation event: sentiment drops by predefined threshold within `n` turns and remains below baseline for at least `m` turns.
- Convergence event: rolling sentiment variance falls below threshold by turn `t`.
- Polarization episode: topic-week exceeds percentile cutoff on bimodality/dispersion metric.

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
- Platform-affordance sensitivity (engagement-controlled subsets).
- Temporal robustness (early vs late period splits).
- Placebo tests for contagion:
  - Future-lag placebo.
  - Shuffled within-thread exposure.

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
- Platform confounds: moderation/UI differences across MoltBook and Reddit/X.

### 16. Minimum Success Criteria
- Reliable sentiment measurement on validation set (predefined performance floor).
- Statistically and practically interpretable AI-vs-human differences.
- At least one robust dynamic finding (trajectory or contagion) surviving sensitivity checks.
- Transparent limitations and failure modes documented.

### 17. Planned Outputs
- Main paper tables:
  - Sentiment distribution by platform/topic.
  - Dynamics and contagion model coefficients.
  - Safety index top-risk topics/time windows.
- Figures:
  - Topic-time heatmap.
  - Thread trajectory archetypes.
  - Comparative distribution overlays.
- Appendix:
  - Annotation guidelines, ablations, placebo tests, diagnostics.
