"""Provisional benchmark of all 6 scorers vs single-rater gold (250) + stats fixes.

Gold caveat: MoltBook_Gold_Annotation_250.xlsx Annotation sheet is rater-1
only (250/250); rater-2/adjudicated are blank. This benchmark is PROVISIONAL
(single-rater reference) until the overlap set is double-annotated and
adjudicated. Do not treat as validation of the main scorer.

Also recomputes: parent-child chi2 (correct df=4) + Cramer's V + permutation
p for alignment; epsilon^2 for Kruskal H values; lift over base rate for
neutral-attractor / de-escalation claims; trend test for early->late positive
accumulation; thread-level bootstrap CIs for key proportions.

Usage: python3 scripts/provisional_benchmark_and_stats.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

LABELS = ["negative", "neutral", "positive"]
L2I = {lab: i for i, lab in enumerate(LABELS)}
REPO = Path(__file__).resolve().parents[1]


def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for i in range(3):
        tp = float(((y_true == i) & (y_pred == i)).sum())
        fp = float(((y_true != i) & (y_pred == i)).sum())
        fn = float(((y_true == i) & (y_pred != i)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s))


def load_gold():
    wb = openpyxl.load_workbook(REPO / "MoltBook_Gold_Annotation_250.xlsx", data_only=True, read_only=True)
    ann = {r[1]: {"r1": r[3], "final": r[10]} for r in wb["Annotation"].iter_rows(min_row=2, values_only=True)}
    sco = {}
    for r in wb["Scorers"].iter_rows(min_row=2, values_only=True):
        sco[r[0]] = {"vader": r[2], "swn": r[3], "ensemble": r[4]}
    wb.close()
    tx = pd.read_csv(REPO / "transformer_labels_250.csv", dtype=str).set_index("comment_id").to_dict("index")
    rows = []
    for cid, a in ann.items():
        gold = (a["final"] or a["r1"] or "").strip().lower()
        if gold not in L2I:
            continue
        rows.append({
            "comment_id": cid, "gold": gold,
            "vader_label": str(sco[cid]["vader"]).strip().lower(),
            "swn_label": str(sco[cid]["swn"]).strip().lower(),
            "ensemble_label": str(sco[cid]["ensemble"]).strip().lower(),
            "twitter_roberta_label": str(tx[cid]["twitter_roberta_label"]).strip().lower(),
            "xlm_twitter_label": str(tx[cid]["xlm_twitter_label"]).strip().lower(),
            "bertweet_label": str(tx[cid]["bertweet_label"]).strip().lower(),
        })
    return pd.DataFrame(rows)


def benchmark(df):
    out = {}
    for col in ["vader_label", "swn_label", "ensemble_label", "twitter_roberta_label",
                "xlm_twitter_label", "bertweet_label"]:
        yt = df["gold"].map(L2I).to_numpy()
        yp = df[col].map(L2I).to_numpy()
        acc = float((yt == yp).mean())
        out[col] = {"n": int(len(df)), "accuracy": round(acc, 4), "macro_f1": round(macro_f1(yt, yp), 4),
                    "counts": Counter(df[col].tolist())}
    return out


def ensemble_tuning():
    """Grid-search (vader_t, swn_t) on stratified dev half; report on held-out test half."""
    rb = pd.read_csv(REPO / "data/rule_based/moltbook_rule_based_comments_20260522T162134Z.csv",
                     usecols=["comment_id", "vader_compound", "swn_score"])
    gold = load_gold()[["comment_id", "gold"]].merge(rb, on="comment_id", how="left")
    assert gold[["vader_compound", "swn_score"]].notna().all().all(), "missing raw scores for gold ids"
    rng = random.Random(42)
    dev_idx, test_idx = [], []
    for lab, grp in gold.groupby("gold"):
        ids = grp["comment_id"].tolist()
        rng.shuffle(ids)
        k = len(ids) // 2
        dev_idx += ids[:k]
        test_idx += ids[k:]
    dev = gold[gold.comment_id.isin(dev_idx)].reset_index(drop=True)
    test = gold[gold.comment_id.isin(test_idx)].reset_index(drop=True)

    def lab(v, t):
        return "positive" if v >= t else ("negative" if v <= -t else "neutral")

    best = None
    for vt in [0.0, 0.05, 0.1, 0.2, 0.3]:
        for st in [0.0, 0.02, 0.05, 0.1]:
            pred = [lab(v, vt) if lab(v, vt) == lab(s, st) else "neutral"
                    for v, s in zip(dev.vader_compound, dev.swn_score)]
            f1 = macro_f1(dev.gold.map(L2I), pd.Series(pred).map(L2I))
            if best is None or f1 > best[0]:
                best = (f1, vt, st)
    f1_dev, bvt, bst = best
    for name, split in [("dev", dev), ("test", test)]:
        pass
    test_pred = [lab(v, bvt) if lab(v, bvt) == lab(s, bst) else "neutral"
                 for v, s in zip(test.vader_compound, test.swn_score)]
    def_pred = [lab(v, 0.05) if lab(v, 0.05) == lab(s, 0.02) else "neutral"
                for v, s in zip(test.vader_compound, test.swn_score)]
    return {
        "dev_n": len(dev), "test_n": len(test),
        "best_dev": {"vader_t": bvt, "swn_t": bst, "macro_f1": round(f1_dev, 4)},
        "default_on_test": {"macro_f1": round(macro_f1(test.gold.map(L2I), pd.Series(def_pred).map(L2I)), 4),
                            "accuracy": round(float((test.gold == def_pred).mean()), 4)},
        "tuned_on_test": {"macro_f1": round(macro_f1(test.gold.map(L2I), pd.Series(test_pred).map(L2I)), 4),
                          "accuracy": round(float((test.gold == test_pred).mean()), 4)},
    }


def stats_fixes():
    res = {}
    # Parent-child contingency (from artifact)
    cont = pd.read_csv(REPO / "data/figures/rq3_contingency_table_20260509T173800Z.csv", index_col=0)
    obs = cont.loc[["negative", "neutral", "positive"], ["negative", "neutral", "positive"]].to_numpy(dtype=float)
    n = obs.sum()
    row_s = obs.sum(axis=1, keepdims=True)
    col_s = obs.sum(axis=0, keepdims=True)
    exp = row_s @ col_s / n
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof = (3 - 1) * (3 - 1)
    from math import erf, exp as mexp, sqrt
    # p via scipy if available else stored artifact p
    try:
        from scipy.stats import chi2 as chi2dist
        p = float(chi2dist.sf(chi2, dof))
    except ImportError:
        p = 0.2869  # artifact value rq3_alignment_stats
    V = float((chi2 / (n * 2)) ** 0.5)
    exact = float(np.trace(obs) / n)
    # permutation test for exact-match rate (child labels permuted)
    rng = np.random.default_rng(42)
    parent = np.repeat([0, 1, 2], obs.sum(axis=1).astype(int))
    child_pool = np.repeat([0, 1, 2], obs.sum(axis=0).astype(int))
    sims = np.array([float((parent == rng.permutation(child_pool)).mean()) for _ in range(2000)])
    p_perm = float(((sims >= exact).sum() + 1) / (len(sims) + 1))
    res["parent_child"] = {"n_pairs": int(n), "chi2": round(chi2, 3), "dof": dof,
                           "p_chi2": round(p, 4), "cramers_V": round(V, 4),
                           "exact_match": round(exact, 4),
                           "permutation_p_gte": round(p_perm, 4)}
    # Lift over base rate for neutral-attractor claims
    base_neu = float(col_s[0, 1] / n)
    res["lift"] = {
        "base_rate_child_neutral": round(base_neu, 4),
        "p_neutral_given_negative_parent": round(float(obs[0, 1] / row_s[0, 0]), 4),
        "lift_neg_to_neu": round(float((obs[0, 1] / row_s[0, 0]) / base_neu), 3),
        "p_neutral_given_neutral_parent": round(float(obs[1, 1] / row_s[1, 0]), 4),
        "lift_neu_to_neu": round(float((obs[1, 1] / row_s[1, 0]) / base_neu), 3),
        "note": "n_negative_parents=75: CIs wide; report with cluster bootstrap",
    }
    # Kruskal epsilon^2 (need N = total obs; use staged n=1296 for corpus-level groupings)
    N = 1296
    for name, H, k in [("post_thread", 110.8733, 47), ("author", 308.7110, 170)]:
        eps2 = (H - k + 1) / (N - k)
        res[f"kruskal_{name}"] = {"H": H, "k_groups": k, "N": N, "epsilon_squared": round(float(eps2), 4)}
    res["kruskal_note"] = "post_id==thread_id 1:1 (55/55); identical H is redundant grouping, not replication"
    # Trajectory trend: early/mid/late positive shares from design matrix table
    # Cochran-Armitage approx via tree counts? Use reported phase shares with equal-n trend test placeholder:
    # Instead do Spearman on per-thread phase order vs positive proportion from artifact.
    traj = pd.read_csv(REPO / "data/figures/rq3_trajectory_by_phase_20260509T173800Z.csv")
    pos = traj[traj.sentiment == "positive"].copy()
    pos["phase_o"] = pos.phase.map({"early": 0, "mid": 1, "late": 2})
    try:
        from scipy.stats import spearmanr
        rho, p_trend = spearmanr(pos.phase_o, pos.proportion)
        res["trend"] = {"spearman_rho_phase_vs_posprop": round(float(rho), 4), "p": round(float(p_trend), 4),
                        "n_thread_phases": int(len(pos))}
    except ImportError:
        res["trend"] = {"means": pos.groupby("phase").proportion.mean().round(4).to_dict(),
                        "note": "scipy missing; report means + add trend test after install"}
    # Thread-level bootstrap CI for corpus positive share (cluster bootstrap over threads)
    staged = pd.read_json(REPO / "data/staged/moltbook_comments_all.jsonl", lines=True)
    # need ensemble labels per comment: join rule-based latest
    rb = pd.read_csv(REPO / "data/rule_based/moltbook_rule_based_comments_20260522T162134Z.csv",
                     usecols=["comment_id", "ensemble_label"])
    m = staged[["comment_id", "thread_id"]].merge(rb, on="comment_id")
    threads = m.groupby("thread_id").ensemble_label.apply(list)
    rng2 = np.random.default_rng(42)
    boots = []
    T = len(threads)
    for _ in range(2000):
        samp = rng2.integers(0, T, T)
        labs = [lab for i in samp for lab in threads.iloc[i]]
        boots.append(sum(l == "positive" for l in labs) / len(labs))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    res["cluster_bootstrap_pos_share"] = {"threads": int(T), "point": round(float((m.ensemble_label == 'positive').mean()), 4),
                                         "ci95": [round(lo, 4), round(hi, 4)]}
    return res


def main():
    df = load_gold()
    bench = benchmark(df)
    tune = ensemble_tuning()
    stats = stats_fixes()
    out = {"n_gold": len(df),
           "gold_dist": dict(Counter(df.gold)),
           "caveat": "PROVISIONAL single-rater reference (rater-1 only, n=250). Kappa not computable; adjudication pending.",
           "benchmark": bench, "ensemble_tuning": tune, "stats": stats}
    # choose main scorer provisionally by macro-F1
    ranked = sorted(bench.items(), key=lambda kv: kv[1]["macro_f1"], reverse=True)
    out["provisional_ranking"] = [[k, v["macro_f1"], v["accuracy"]] for k, v in ranked]
    p = REPO / "data/gold/provisional_benchmark_250.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
