"""Stage-7 selection on synthetic data (docs/stage7_plan.md §5).

T-47: train rows only; the allow-list refuses a leaking column.
T-48: date folds, the kW score, the selection model per route (R7-04), the stage-9 input per
      target × horizon (R7-04, F-29), correlated pairs, RFE keeps the core, the elbow rule,
      stability on whole days, the MI vote against shuffled copies, consensus and 05f tiers.
T-49: the k history follows the stage-6 rules (DST guard, lags per horizon, past only).
"""

import numpy as np
import pandas as pd
import pytest

from solarpv.config.constants import AC_CAPACITY_KW
from solarpv.selection import common as C
from solarpv.selection import steps as S


def synthetic(days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-05-01", periods=24 * days, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    sun = np.clip(np.sin((hour - 4) / 14 * np.pi), 0, None)
    signal = rng.uniform(0.3, 1.0, len(idx))
    t = pd.DataFrame(index=idx)
    t["split"] = "train"
    t["warmup"] = False
    t["op_limited_flag"] = False
    t["daytime"] = sun > 0.05
    t["theo_kw"] = 3000 * sun
    t["k_valid"] = t["daytime"]
    t["signal"] = signal
    t["signal_copy"] = 2 * signal + 1e-6 * rng.normal(size=len(idx))
    t["noise"] = rng.normal(size=len(idx))
    t["aoi"] = rng.normal(size=len(idx))  # a core name carrying no information
    t["clear"] = t["theo_kw"]  # what the direct target needs besides k
    t["target_k"] = np.where(t["daytime"], signal + 0.02 * rng.normal(size=len(idx)), np.nan)
    t["site_total_AC_kW"] = np.clip(t["target_k"].fillna(0) * t["theo_kw"], 0, None)
    return t


# ── T-47 ──────────────────────────────────────────────────────────────────


def test_selection_refuses_rows_outside_train():
    t = synthetic()
    t.iloc[-24:, t.columns.get_loc("split")] = "val"
    with pytest.raises(ValueError, match="train rows only"):
        C.check_train_only(t)


def test_pool_is_an_allow_list_and_refuses_leaking_columns():
    t = synthetic()
    with pytest.raises(ValueError):
        C.pool(t, 1)  # the synthetic table lacks the contract: nothing outside it passes
    t2 = pd.DataFrame(columns=["s01_inv1_DC_kW"])
    with pytest.raises(ValueError):
        C.pool(t2, 1, ("s01_inv1_DC_kW",))


# ── T-48 ──────────────────────────────────────────────────────────────────


def test_date_folds_expand_keep_a_gap_and_never_overlap():
    p = C.problem(synthetic(days=60), "indirect")
    folds = C.date_folds(p.day, n_folds=5, gap_days=3)
    assert len(folds) == 5
    last_test_end = None
    for train, test in folds:
        assert not (train & test).any()
        train_days, test_days = p.day[train].unique(), p.day[test].unique()
        assert (train_days.max() + pd.Timedelta(days=3)) < test_days.min()
        if last_test_end is not None:
            assert test_days.min() > last_test_end
        last_test_end = test_days.max()


def test_score_is_in_kw_for_both_targets():
    t = synthetic()
    for target in ("indirect", "direct"):
        p = C.problem(t, target)
        score, shap = C.cv_score(p, ["signal", "noise", "clear"])
        assert 0 < score < 10  # % of capacity, a good model on a clean signal
        assert shap["signal"] > 5 * shap["noise"]
    p = C.problem(t, "indirect")
    rows = p.score
    kw = C.to_kw(p, np.full(int(rows.sum()), 2.0), rows)
    assert kw.max() <= AC_CAPACITY_KW  # k x theo, clipped to the capacity


def test_selection_model_is_the_fixed_xgboost_for_both_routes(monkeypatch):
    """R7-04: both routes are selected with the fixed XGBoost of D-11; an explicit model still
    wins; the thread count is fixed (RH-03), never "all cores"."""
    from solarpv.config.constants import XGB_THREADS
    from solarpv.features.decide_k import XGB_PARAMS

    seen, real = [], C.model
    monkeypatch.setattr(C, "model", lambda params, seed: seen.append(params) or real(params, seed))
    t = synthetic(days=30)
    for target in ("indirect", "direct"):
        seen.clear()
        C.cv_score(C.problem(t, target), ["signal", "clear"], with_shap=False)
        assert seen and all(p is XGB_PARAMS for p in seen)
    seen.clear()
    C.cv_score(C.problem(t, "direct"), ["signal", "clear"], C.SELECTION_XGB, with_shap=False)
    assert seen and all(p is C.SELECTION_XGB for p in seen)
    assert XGB_PARAMS["n_jobs"] == C.SELECTION_XGB["n_jobs"] == XGB_THREADS > 0


def test_model_input_follows_the_registered_steps():
    """R7-04 steps 3 and 4 (F-29): a failed inside-train check gives the full pool, and
    validation cannot overturn it; a passed check gives the 05e set unless validation raises a
    red flag."""
    from solarpv.selection.run import decide_input

    assert decide_input(False, None)[0] == "full pool"
    assert decide_input(False, False)[0] == "full pool"  # validation is not used
    assert decide_input(False, True)[0] == "full pool"
    assert decide_input(True, None)[0] == "pending"
    assert decide_input(True, True)[0] == "full pool"
    assert decide_input(True, False)[0] == "05e"


def test_correlated_pairs_are_reported_without_chaining():
    t = synthetic()
    t["between"] = t["signal_copy"] + 0.35 * t["noise"].abs()  # ~ signal_copy, not ~ noise
    pairs = S.correlated_pairs(t, ["signal", "signal_copy", "noise", "between"], t["daytime"])
    assert [(a, b) for a, b in zip(pairs["a"], pairs["b"], strict=True)] == [
        ("signal", "signal_copy")]  # pairs only: no group, nothing removed (R7-01)


def test_rfe_removes_one_copy_of_a_duplicate_pair():
    p = C.problem(synthetic(), "indirect")
    kept, _ = S.rfe(p, ["signal", "signal_copy", "noise", "aoi"], protected={"aoi"}, target_n=2)
    assert "aoi" in kept and len({"signal", "signal_copy"} & set(kept)) == 1


def test_rfe_never_drops_the_core():
    p = C.problem(synthetic(), "indirect")
    kept, history = S.rfe(p, ["signal", "noise", "aoi"], protected={"aoi"}, target_n=2)
    assert set(kept) == {"signal", "aoi"}
    assert history["removed"].iloc[0] == "noise"


def test_elbow_rule_is_relative_to_the_starting_set():
    curve = pd.DataFrame({"n_features": [4, 3, 2, 1], "score": [10.0, 10.04, 10.08, 12.0],
                          "features": ["a|b|c|d", "a|b|c", "a|b", "a"]})
    e = S.elbows(curve)
    assert e["0.5pct"] == 3 and e["1.0pct"] == 2 and e["2.0pct"] == 2
    assert S.elbow_set(curve) == ["a", "b"]


def test_stability_uses_whole_days_and_finds_the_signal():
    p = C.problem(synthetic(), "indirect")
    prob = S.stability(p, ["signal", "noise"], n_sub=4, share=0.2)
    assert prob["signal"] == 1.0
    assert prob["noise"] < S.STABILITY_VOTE


def test_mi_vote_rejects_noise_and_accepts_signal():
    t = synthetic()
    p = C.problem(t, "indirect")
    mi = S.mutual_information(p.X, p.y, ["signal", "noise"], p.fit)
    votes, threshold = S.mi_vote(mi, p.X, p.y, p.fit)
    assert bool(votes["signal"]) and not bool(votes["noise"])
    assert mi["noise"] <= threshold < mi["signal"]


def test_consensus_and_tiers_follow_the_rules():
    votes = pd.DataFrame({"a": [True, True, False], "b": [True, False, False],
                          "c": [True, True, True]}, index=["shap", "stab", "mi"]).T
    assert S.consensus(votes) == ["c", "a"]
    prob = pd.Series({"x": 0.9, "y": 0.6, "z": 0.4, "w": 0.4, "v": 0.1, "u": 0.9})
    mi = pd.Series({"x": 1.0, "y": 1.0, "z": 2.0, "w": 0.1, "v": 1.0, "u": 1.0})
    n = pd.Series({"x": 2, "y": 2, "z": 1, "w": 1, "v": 1, "u": 0})
    tier = S.tiers(prob, mi, n).set_index("feature")["tier"]
    assert tier.to_dict() == {"x": "CORE", "y": "SUPPORTED", "z": "CONDITIONAL",
                              "w": "EXCLUDED", "v": "EXCLUDED", "u": "EXCLUDED"}


# ── T-49 ──────────────────────────────────────────────────────────────────


def k_table() -> pd.DataFrame:
    idx = pd.date_range("2023-03-10", "2023-04-10", freq="h", tz="UTC", inclusive="left")
    t = pd.DataFrame(index=idx)
    t["k_valid"] = True
    t["theo_kw"] = 500.0
    t["op_limited_flag"] = False
    t["target_k"] = np.arange(len(idx), dtype=float)
    return t


def test_k_history_lags_follow_the_horizon_and_the_dst_guard():
    t = k_table()
    for h, expected in ((1, {1, 2, 3, 7, 14}), (2, {2, 3, 7, 14}), (3, {3, 7, 14})):
        k = C.k_history(t, h)
        lags = {int(c.split("_")[3][:-1]) for c in k.columns if "_lag_" in c}
        assert lags == expected
    k1 = C.k_history(t, 1)
    row = pd.Timestamp("2023-03-20 10:00", tz="UTC")
    assert k1.loc[row, "target_k_lag_1d"] == t.loc[row - pd.Timedelta(hours=24), "target_k"]
    # on the night of the DST change (2023-03-24 local) an hour 24 h back may end after T0
    t0 = pd.Series(pd.DatetimeIndex(C.build.issuance_utc(t.index, 1)), index=t.index)
    source_end = t.index - pd.Timedelta(hours=23)
    guarded = source_end > t0.to_numpy()
    raw = C.build.past_value(t["target_k"], 1, C.build.issuance_utc(t.index, 1))
    assert np.isnan(raw[guarded]).all()


def test_k_history_uses_only_the_past():
    t = k_table()
    row = pd.Timestamp("2023-03-28 09:00", tz="UTC")
    before = C.k_history(t, 1).loc[row]
    t2 = t.copy()
    t2.loc[t2.index >= C.build.issuance_utc(pd.DatetimeIndex([row]), 1)[0], "target_k"] = -99.0
    after = C.k_history(t2, 1).loc[row]
    pd.testing.assert_series_equal(before, after)


def test_k_history_ignores_invalid_hours():
    t = k_table()
    t["op_limited_flag"] = True
    assert C.k_history(t, 1).isna().all().all()
