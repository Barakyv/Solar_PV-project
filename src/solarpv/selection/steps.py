"""The steps of the stage-7 selection (docs/stage7_plan.md §3–4). Pure functions on a Problem.

Base stage (as in the January base notebooks that fed the submission): variance and MI filters
with the core exempt, duplicate removal, RFE to 42 with the core kept. Then 05c (SHAP backward
elimination), 05e (three votes, 2 of 3) and 05f (tiered audit). Deviations from the original are
marked P<n> (plan §2a).
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from solarpv.selection.common import SEED, Problem, cv_score

VARIANCE_MIN = 0.001
MI_MIN = 0.02
MI_NEIGHBOURS = 5
DUPLICATE_R = 0.98
RFE_TARGET = 42
ELBOW_THRESHOLDS = (0.005, 0.01, 0.02)
ELBOW_USED = 0.01
STABILITY_SUBSAMPLES = 30
STABILITY_FRACTION = 0.7
STABILITY_SHARE = 0.01
STABILITY_VOTE = 0.6
MI_SHUFFLES = 3
MI_NULL_QUANTILE = 0.95
TIER_CORE, TIER_SUPPORTED, TIER_FLOOR = 0.75, 0.50, 0.30
FLOORS = (0.25, 0.30, 0.35, 0.40)


# ── base stage ────────────────────────────────────────────────────────────


def low_variance(X: pd.DataFrame, feats: list[str], protected: set[str]) -> list[str]:
    var = X[feats].var()
    return [f for f in feats if var[f] < VARIANCE_MIN and f not in protected]


def complete_rows(X: pd.DataFrame, feats: list[str], rows: pd.Series) -> pd.Series:
    """Rows of `rows` with no missing value in `feats` (P3: no filling here)."""
    return rows & X[feats].notna().all(axis=1)


def mutual_information(X: pd.DataFrame, y: pd.Series, feats: list[str], rows: pd.Series,
                       seed: int = SEED) -> pd.Series:
    use = complete_rows(X, feats, rows) & y.notna()
    mi = mutual_info_regression(X.loc[use, feats].to_numpy(), y[use].to_numpy(),
                                n_neighbors=MI_NEIGHBOURS, random_state=seed)
    return pd.Series(mi, index=feats)


def low_mi(mi: pd.Series, protected: set[str]) -> list[str]:
    return [f for f in mi.index if mi[f] < MI_MIN and f not in protected]


def duplicate_groups(X: pd.DataFrame, feats: list[str], rows: pd.Series) -> list[list[str]]:
    """Connected components of |Pearson r| > 0.98 on the selection rows (B4)."""
    corr = X.loc[complete_rows(X, feats, rows), feats].corr().abs()
    parent = {f: f for f in feats}

    def root(f: str) -> str:
        while parent[f] != f:
            f = parent[f]
        return f

    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            if corr.loc[a, b] > DUPLICATE_R:
                parent[root(b)] = root(a)
    groups: dict[str, list[str]] = {}
    for f in feats:
        groups.setdefault(root(f), []).append(f)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def deduplicate(groups: list[list[str]], mi: pd.Series, core: set[str]) -> dict[str, str]:
    """{removed feature: kept representative}. A core member wins its group; otherwise the
    highest MI (ties by name)."""
    removed = {}
    for members in groups:
        ranked = sorted(members, key=lambda f: (f not in core, -mi[f], f))
        for f in ranked[1:]:
            removed[f] = ranked[0]
    return removed


def rfe(problem: Problem, feats: list[str], protected: set[str], target_n: int = RFE_TARGET,
        log=None) -> tuple[list[str], pd.DataFrame]:
    """Remove one feature per step, the lowest mean |SHAP| on the held-out folds outside the
    protected set (P7), until `target_n` remain. History: one row per step."""
    current, rows = list(feats), []
    while len(current) > target_n:
        score, shap = cv_score(problem, current)
        candidates = shap.drop(index=[f for f in current if f in protected])
        if candidates.empty:
            break
        drop = candidates.sort_values(kind="stable").index[0]
        rows.append({"n_features": len(current), "score": score, "removed": drop,
                     "removed_shap": float(shap[drop])})
        current.remove(drop)
        if log and len(rows) % 10 == 0:
            log.note(f"  RFE {len(current)} left, score {score:.3f}")
    score, _ = cv_score(problem, current, with_shap=False)
    rows.append({"n_features": len(current), "score": score, "removed": "", "removed_shap": np.nan})
    return current, pd.DataFrame(rows)


# ── 05c ───────────────────────────────────────────────────────────────────


def elimination(problem: Problem, feats: list[str]) -> pd.DataFrame:
    """SHAP backward elimination from `feats` to one feature, refitting at each size. Row n holds
    the score of the n-feature set and the feature removed next."""
    current, rows = list(feats), []
    while current:
        score, shap = cv_score(problem, current)
        drop = shap.sort_values(kind="stable").index[0]
        rows.append({"n_features": len(current), "score": score,
                     "features": "|".join(current), "removed_next": drop})
        current.remove(drop)
    return pd.DataFrame(rows)


def elbows(curve: pd.DataFrame) -> dict[str, int]:
    """Smallest set whose score is at most t (relative) above the full set's, for every
    threshold; and the size at the largest second difference of the curve."""
    c = curve.sort_values("n_features")
    base = float(c["score"].iloc[-1])
    out = {f"{100 * t:.1f}pct": int(c.loc[c["score"] <= base * (1 + t), "n_features"].min())
           for t in ELBOW_THRESHOLDS}
    second = c["score"].to_numpy()
    if len(second) >= 3:
        d2 = second[:-2] - 2 * second[1:-1] + second[2:]
        out["2nd_deriv"] = int(c["n_features"].iloc[1 + int(np.argmax(d2))])
    return out


def elbow_set(curve: pd.DataFrame, threshold: float = ELBOW_USED) -> list[str]:
    n = elbows(curve)[f"{100 * threshold:.1f}pct"]
    return curve.loc[curve["n_features"] == n, "features"].iloc[0].split("|")


# ── 05e ───────────────────────────────────────────────────────────────────


def stability(problem: Problem, feats: list[str], n_sub: int = STABILITY_SUBSAMPLES,
              fraction: float = STABILITY_FRACTION, share: float = STABILITY_SHARE,
              seed: int = SEED) -> pd.Series:
    """Probability that a feature holds at least `share` of the total mean |SHAP| on the
    held-out days, over subsamples of `fraction` of the days (P9)."""
    rng = np.random.default_rng(seed)
    days = np.unique(problem.day[problem.fit | problem.score].to_numpy())
    hits = pd.Series(0.0, index=feats)
    for i in range(n_sub):
        chosen = rng.choice(days, size=int(round(fraction * len(days))), replace=False)
        inside = problem.day.isin(chosen)
        _, shap = cv_score(problem, feats, seed=seed + i, folds=[(inside, ~inside)])
        hits += (shap / shap.sum() >= share).astype(float)
    return hits / n_sub


def mi_vote(mi: pd.Series, X: pd.DataFrame, y: pd.Series, rows: pd.Series,
            seed: int = SEED) -> tuple[pd.Series, float]:
    """Vote when the MI exceeds the 95th percentile of the MI of shuffled copies of every
    feature (a pooled null; P8). Returns (votes, threshold)."""
    feats = list(mi.index)
    use = complete_rows(X, feats, rows) & y.notna()
    rng = np.random.default_rng(seed)
    values = X.loc[use, feats].to_numpy()
    null = []
    for r in range(MI_SHUFFLES):
        shuffled = np.column_stack([rng.permutation(values[:, j]) for j in range(len(feats))])
        null.extend(mutual_info_regression(shuffled, y[use].to_numpy(),
                                           n_neighbors=MI_NEIGHBOURS, random_state=seed + r))
    threshold = float(np.quantile(null, MI_NULL_QUANTILE))
    return mi > threshold, threshold


def consensus(votes: pd.DataFrame) -> list[str]:
    """Features with at least 2 of the 3 votes, in order of the number of votes."""
    n = votes.sum(axis=1)
    return list(n[n >= 2].sort_values(ascending=False, kind="stable").index)


# ── 05f ───────────────────────────────────────────────────────────────────


def tiers(prob: pd.Series, mi: pd.Series, n_votes: pd.Series, floor: float = TIER_FLOOR
          ) -> pd.DataFrame:
    """The tiers of the submitted 05f files: CORE ≥ 0.75, SUPPORTED ≥ 0.50, CONDITIONAL ≥ floor
    with MI above its 75th percentile, otherwise EXCLUDED (also with no vote)."""
    q75 = float(mi.quantile(0.75))
    rows = []
    for f in prob.index:
        p, m = float(prob[f]), float(mi[f])
        if n_votes[f] == 0:
            tier, reason = "EXCLUDED", "insufficient_votes (0/3)"
        elif p >= TIER_CORE:
            tier, reason = "CORE", ""
        elif p >= TIER_SUPPORTED:
            tier, reason = "SUPPORTED", ""
        elif p >= floor and m > q75:
            tier, reason = "CONDITIONAL", ""
        elif p >= floor:
            tier, reason = "EXCLUDED", f"unstable_and_weak_MI (stab={p:.2f}, MI={m:.3f})"
        else:
            tier, reason = "EXCLUDED", f"stability_too_low ({p:.2f} < {floor:.2f} floor)"
        rows.append({"feature": f, "stability": p, "mi": m, "votes": int(n_votes[f]),
                     "tier": tier, "reason": reason})
    return pd.DataFrame(rows).assign(mi_75th=q75)


def floor_sensitivity(prob: pd.Series, mi: pd.Series, n_votes: pd.Series) -> pd.DataFrame:
    rows = []
    for floor in FLOORS:
        t = tiers(prob, mi, n_votes, floor)
        rows.append({"floor": floor, **t["tier"].value_counts().to_dict(),
                     "recommended": int((t["tier"] != "EXCLUDED").sum())})
    return pd.DataFrame(rows).fillna(0)
