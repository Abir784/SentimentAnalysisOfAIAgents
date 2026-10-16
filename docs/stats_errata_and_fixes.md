# Stats errata and fixes (2026-10-16)

Recomputed from tracked artifacts; script: `scripts/provisional_benchmark_and_stats.py`.
Full numbers: `data/gold/provisional_benchmark_250.json` → `.stats`.

## 1. Parent-child χ²: df 2 → 4
- Code (`run_moltbook_rq3_sentiment_dynamics.py:134`) runs `chi2_contingency`
  on a 3×3 table; correct `dof = (3−1)(3−1) = 4`. The `df=2` in the text had
  no source (dof is computed then discarded, never saved).
- Recomputed: χ² = 5.004, df = 4, p = 0.287 (unchanged conclusion),
  Cramér's V = 0.045 (negligible), exact-match 48.1%.
- Permutation test (permute child labels, 2000 reps, seed 42):
  p = 0.087 for ≥ observed exact-match. Wording: "**no evidence of
  dependence**", not "no contagion".
- Caveat: pairs are sequential-adjacency proxies, not resolved `parent_id`
  links. Rerun on direct edges only before any causal claim.

## 2. Posts vs threads identical H
- `post_id == thread_id` 1:1 (55/55), so H = 110.87 twice is one grouping
  reported twice. Collapsed to a single row.
- Effect sizes (ε² = (H−k+1)/(N−k), N = 1296): post/thread 0.052 (small),
  author (H = 308.71, k = 170) 0.124. Compare ε², never raw H across
  different k.

## 3. Attractor / de-escalation → lift over base rate
- Child base rate neutral: 56.2%. P(neutral | negative parent) = 62.7%
  (lift 1.11); P(neutral | neutral parent) = 57.3% (lift 1.02).
- n_negative_parents = 75 → wide CIs. "De-escalation" downgraded to a weak,
  unconfirmed signal; "attractor" to a descriptive plurality. Both need
  thread-cluster bootstrap CIs in the full paper.

## 4. "29.2% / 63.6%" are early-phase starting values, not changes
- Phase means (per-thread proportions): early 29.1% pos / 63.6% neu(implied),
  mid 34.3%, late 36.4%. Drift +7.3pp positive is descriptive.
- Spearman trend (phase order vs per-thread positive share, n = 141):
  ρ = 0.08, p = 0.32 — not significant. No "accumulation" claim.

## 5. Uniformity χ² dropped
- χ² = 450.63 vs uniform-thirds is uninformative (any realistic corpus
  rejects it). Distribution now reported with CIs only, plus a
  thread-cluster bootstrap for the positive share: 39.1% [33.4%, 44.9%].
- Code untouched (`run_moltbook_rq2_reporting.py:125`); claims removed from
  abstract-facing text.

## 6. Reply links: direct vs sequential
- RQ3 pairs and RQ1 graph use sequential proxies (`thread_depth` order),
  labeled `sequential_fallback` / `fallback_triggered: True`.
- Required follow-up: count direct (`parent_id → comment_id`) vs sequential
  edges; rerun parent-child analysis on direct-only edges.
