"""T-50: the stage-7 outputs (outputs/07_feature_selection/, tracked in git).

Twelve selection files in the schema of the submitted ones; every 05e feature comes from the
allow-list of its pool and passes the leakage rules; the selection read train rows only; the
k-history decision follows the D-24 rule; the selection model of each route is recorded and the
inside-train check follows the margin rule (R7-04); the stage-9 input per target × horizon
follows the registered steps (R7-04, F-29); the validation comparison is present.
"""

import json

import pytest
import yaml

from solarpv.config.paths import CONFIGS_DIR, OUTPUTS_DIR
from solarpv.features.contract import features_for_horizon
from solarpv.features.leakage import is_leaky

OUT = OUTPUTS_DIR / "07_feature_selection"
COMBOS = [(t, h) for t in ("indirect", "direct") for h in (1, 2, 3)]


def _json(name: str) -> dict:
    path = OUT / name
    if not path.exists():
        pytest.skip("run: python -m solarpv.selection.run")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("target", "horizon"), COMBOS)
def test_05e_file_schema_and_allow_list(target, horizon):
    e = _json(f"final_features_{target}_D{horizon}.json")
    assert e["target_label"] == target and e["horizon"] == horizon
    assert set(e["methods"]) == {"shap_elbow", "stability_selection", "mutual_information"}
    chosen = e["recommended"]["features"]
    assert chosen == e["consensus"]["features"] and len(chosen) == e["recommended"]["n_features"]
    assert all(e["consensus"]["votes"][f] >= 2 for f in chosen)
    start = e["methods"]["shap_elbow"]["shap_rank"]
    assert set(chosen) <= set(start) and len(start) == e["n_starting_features"]
    assert not [f for f in chosen if is_leaky(f)]
    allowed_extra = ("fcst_gfs_", "irradiance_clear_index_prevday")
    allowed_extra += ("target_k_",) if target == "indirect" else ()
    for f in chosen:
        assert f in features_for_horizon(horizon) or f.startswith(allowed_extra), f
    if horizon > 1:  # lags not known at T0 never appear
        assert not [f for f in start if f.endswith("_lag_1d")]


@pytest.mark.parametrize(("target", "horizon"), COMBOS)
def test_05f_file_is_an_audit_of_the_same_set(target, horizon):
    f = _json(f"refined_features_{target}_D{horizon}_05f.json")
    e = _json(f"final_features_{target}_D{horizon}.json")
    assert f["05e_recommended_n"] == e["recommended"]["n_features"]
    n = sum(t["n"] for t in f["tiers"].values())
    assert n == e["n_starting_features"]
    kept = f["tiers"]["CORE"]["features"] + f["tiers"]["SUPPORTED"]["features"] \
        + f["tiers"]["CONDITIONAL"]["features"]
    assert sorted(kept) == sorted(f["recommended_05f"]["features"])


def test_manifest_train_only_and_decisions():
    m = _json("selection_manifest.json")
    assert m["splits_read_for_selection"] == ["train"]
    assert m["validation_read"] == "S8 only"
    decision = yaml.safe_load((CONFIGS_DIR / "k_history_decision.yaml").read_text(
        encoding="utf-8"))
    assert decision["admitted"] == all(v["passes"] for v in decision["results"].values())
    for name, v in decision["results"].items():
        rule = v["gain_pp"] > v["margin_pp"] if name.endswith("D1") else \
            v["gain_pp"] > -v["margin_pp"]
        assert v["passes"] == rule, name
    for combo in [f"{t}_D{h}" for t, h in COMBOS]:
        assert combo in m["counts"] and combo in m["validation_red_flags"]


@pytest.mark.parametrize(("target", "horizon"), COMBOS)
def test_selection_model_recorded_per_route(target, horizon):
    """R7-04: both routes were selected with the fixed XGBoost of D-11."""
    e = _json(f"final_features_{target}_D{horizon}.json")
    model = e["selection_model"]
    assert (model["n_estimators"], model["max_depth"], model["learning_rate"]) == (400, 6, 0.05)


def test_inner_train_check_follows_the_margin_rule():
    import pandas as pd

    path = OUT / "inner_train_check.csv"
    if not path.exists():
        pytest.skip("run: python -m solarpv.selection.run --skip-validation")
    c = pd.read_csv(path)
    assert len(c) == 12 and set(c["sample"]) == {"date folds", "inner hold-out"}
    assert (abs(c["05e_minus_full_pp"] - (c["nrmse_05e"] - c["nrmse_full"])) < 1e-9).all()
    assert (c["passes"] == (c["05e_minus_full_pp"] <= c["margin_pp"])).all()
    assert (c["margin_pp"] >= 0.05).all()
    m = _json("selection_manifest.json")
    for (t, h), g in c.groupby(["target", "horizon"]):
        assert m["inner_train_passes"][f"{t}_D{h}"] == bool(g["passes"].all())


def test_model_inputs_follow_the_inside_train_check_and_validation():
    """F-29: model_inputs.json, the stage-9 input of every target × horizon, follows R7-04
    steps 3 and 4: the 05e set or the full pool, never pending once S8 has run."""
    import pandas as pd

    from solarpv.selection.run import decide_input

    inputs = _json("model_inputs.json")
    inner = pd.read_csv(OUT / "inner_train_check.csv")
    val_path = OUT / "validation_check.csv"
    val = pd.read_csv(val_path) if val_path.exists() else None
    m = _json("selection_manifest.json")
    assert set(inputs) == {f"{t}_D{h}" for t, h in COMBOS}
    for t, h in COMBOS:
        v = inputs[f"{t}_D{h}"]
        g = inner[(inner["target"] == t) & (inner["horizon"] == h)]
        flag = None if val is None else bool(val.loc[(val["target"] == t)
                                                     & (val["horizon"] == h), "red_flag"].any())
        assert v["source"] == decide_input(bool(g["passes"].all()), flag)[0]
        assert m["model_input"][f"{t}_D{h}"] == v["source"]
        assert len(v["features"]) == v["n_features"] == len(set(v["features"]))
        assert not [f for f in v["features"] if is_leaky(f)]
        if v["source"] == "05e":
            e = _json(f"final_features_{t}_D{h}.json")
            assert v["features"] == e["recommended"]["features"]
        elif v["source"] == "full pool":
            assert v["n_features"] == int(g["n_full"].iloc[0])
        if val is not None:
            assert v["source"] != "pending"


def test_validation_comparison_and_availability_file():
    import pandas as pd

    val = OUT / "validation_check.csv"
    avail = OUT / "feature_availability_classification.csv"
    if not val.exists():
        pytest.skip("run: python -m solarpv.selection.run")
    v = pd.read_csv(val)
    assert set(v["features"]) == {"05e", "full pool"} and len(v) == 12
    m = _json("selection_manifest.json")
    for (t, h), g in v.groupby(["target", "horizon"]):
        score = g.set_index("features")["nrmse_kw_mean"]
        gap = float(g["05e_minus_full_pp"].iloc[0])
        assert abs(gap - (score["05e"] - score["full pool"])) < 1e-9  # sign (R7-02)
        assert g["red_flag"].iloc[0] == (gap > g["margin_pp"].iloc[0])
        assert m["validation_red_flags"][f"{t}_D{h}"] == bool(g["red_flag"].iloc[0])
    a = pd.read_csv(avail)
    assert set(a["availability"]) <= {"FORECAST_SAFE", "HORIZON_DEPENDENT", "PIPELINE_DEPENDENT"}
