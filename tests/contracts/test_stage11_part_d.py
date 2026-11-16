# ruff: noqa: E501  (assert messages; see test_stage11_part_c)
"""Stage 11, part D (docs/stage11_plan.md section 7, OP-69): the quantile crossing in the report.

T-191 checks the arithmetic of `analyses11.crossing_table` on a frame whose answers are worked out by
hand; T-192 checks that the written table is on the scored hours, that the report and the figure give
the numbers of the table, and that the two sources are cited, listed and recorded with their DOIs.
"""

import json

import numpy as np
import pandas as pd
import pytest

from solarpv.config.paths import REPO_ROOT
from solarpv.evaluation import quantiles as Q
from solarpv.figures import analyses11 as A

OUT = REPO_ROOT / A.OUT
DOCS = REPO_ROOT / "docs"
KEYS = A.CROSS_KEYS


def _frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = [  # q0.1, q0.5, q0.9, target, theo (kW), observed (kW)
        (10, 20, 30, "direct", 1000, 25),  # in order
        (10, 40, 30, "direct", 1000, 35),  # crossed; sorted median 30, the point 40: distance 10
        (50, 20, 30, "direct", 1000, 25),  # crossed; sorted median 30, the point 20: distance 10
        (30, 20, 10, "direct", 1000, 15),  # crossed, but the median stays 20
        (0.2, 0.5, 0.4, "indirect", 1000, 450),  # crossed; 500 against a sorted median of 400
        (-0.1, -0.2, 0.3, "indirect", 1000, 100),  # crossed in k; both clip to 0 kW, so the median stays
    ]
    q = pd.DataFrame(rows, columns=["q0.1", "q0.5", "q0.9", "target", "theo", "y"])
    q = q.assign(model="tft_96", seed=42, horizon=1, split="val")
    return q.drop(columns=["theo", "y"]), q["theo"].to_numpy(float), q["y"].to_numpy(float)


def test_t191_crossing_shares_distances_and_losses_on_a_worked_frame():
    q, theo, y = _frame()
    t = A.crossing_table(q, theo, y).set_index("target")
    d, i = t.loc["direct"], t.loc["indirect"]
    assert (d["n"], i["n"], d["n_quantiles"]) == (4, 2, 3)
    assert d["crossed_pct"] == pytest.approx(75.0) and i["crossed_pct"] == pytest.approx(100.0)
    assert d["point_not_median_pct"] == pytest.approx(50.0) and i["point_not_median_pct"] == pytest.approx(50.0)
    assert (d["gap_mean_kw"], d["gap_max_kw"]) == pytest.approx((10.0, 10.0))
    assert (i["gap_mean_kw"], i["gap_max_kw"]) == pytest.approx((100.0, 100.0))
    # the losses are those of quantiles.pinball on the frames in kW, raw and sorted per row
    kw = q[q["target"] == "direct"][["q0.1", "q0.5", "q0.9"]].reset_index(drop=True)
    assert d["pinball_raw_kw"] == pytest.approx(Q.pinball(y[:4], kw))
    assert d["pinball_sorted_kw"] == pytest.approx(Q.pinball(y[:4], Q.ordered(kw)))
    assert d["pinball_sorted_kw"] < d["pinball_raw_kw"]  # sorting lowers the loss where the order matters
    # a factor that is not positive would change the order: refused
    with pytest.raises(AssertionError):
        A.crossing_table(q, np.where(q["target"] == "indirect", 0.0, 1000.0), y)


def _trace() -> pd.Series:
    tr = pd.read_csv(DOCS / "traceability.csv", dtype=str, keep_default_na=False)
    return tr.set_index("claim")["value_new"]


def test_t192_the_table_the_report_the_figure_and_the_sources_agree():
    man = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert f"{A.OUT}/quantile_crossing.csv" in man["outputs_sha256"]
    assert all(rel in man["inputs_sha256"] for *_, rel in A.quantile_files())
    t = pd.read_csv(OUT / "quantile_crossing.csv")
    m = pd.read_csv(OUT / "metrics_kw.csv")
    n = m[m["view"] == "all"].set_index(KEYS)["n"]
    assert (t.set_index(KEYS)["n"] == n.reindex(t.set_index(KEYS).index)).all(), "the scored hours"
    assert set(t["model"]) == {"lstm_sota", "tft_96", "tft_168", "tft"}
    assert (t[t["model"] == "lstm_sota"]["crossed_pct"] == 0).all()
    assert (t["pinball_sorted_kw"] <= t["pinball_raw_kw"] + 1e-9).all()

    # the report gives the numbers of the table (mean over the horizons per model and route)
    tfts = t[t["model"].isin(["tft_96", "tft_168", "tft"])]
    per = tfts.groupby(["model", "target", "split"])["crossed_pct"].mean()
    tr = _trace()
    for split, name in (("val", "validation"), ("test", "test")):
        s = per.xs(split, level="split")
        assert tr[f"quantile outputs: share of hours crossed, TFTs, {name}, highest model and route, %"] == f"{s.max():.1f}"
        assert tr[f"quantile outputs: share of hours crossed, TFTs, {name}, lowest model and route, %"] == f"{s.min():.1f}"
    report = (DOCS / "report_v7.md").read_text(encoding="utf-8")
    assert "### 6.5 Quantile outputs" in report
    figs = json.loads((REPO_ROOT / "outputs/figures/figures_manifest.json").read_text(encoding="utf-8"))["figures"]
    facts = figs["S11-21"]["facts"]
    assert f"{facts['v_hi']:.1f}" == tr["quantile outputs: share of hours crossed, TFTs, validation, highest model and route, %"]
    assert f"{facts['g_t']:.1f}" == tr["quantile outputs: mean distance between the point and the sorted median on those hours, test, kW"]

    # the two sources: cited in the text, listed, and recorded with the DOIs that were checked
    body, refs = report.split("## 17. References", 1)
    for cite, entry in (("Chernozhukov et al., 2010", "Quantile and probability curves without crossing. Econometrica, 78(3), 1093–1125."),
                        ("Koenker & Bassett, 1978", "Regression quantiles. Econometrica, 46(1), 33–50.")):
        assert f"[{cite}]" in body, cite
        assert entry in refs, entry
    plan = (DOCS / "stage11_plan.md").read_text(encoding="utf-8")
    assert "doi:10.3982/ECTA7880" in plan and "doi:10.2307/1913643" in plan
