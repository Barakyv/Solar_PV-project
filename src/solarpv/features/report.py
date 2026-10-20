"""Figures and tables of stage 6 for notebooks/04b_features_target.ipynb.

Each function reads the stage-6 output and returns (table, figure, facts) as in the stage-3b
analyses (solarpv.eda.quality.common.Result), so that the notebook shows the data and the
builder formats the notes from the facts. Train only where a statistic could inform a choice.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarpv.config.constants import AC_CAPACITY_KW, TIMEZONE_LOCAL
from solarpv.config.paths import OUTPUTS_DIR, PROCESSED_DIR
from solarpv.eda.quality.common import Result
from solarpv.features.contract import features_for_horizon

STAGE = PROCESSED_DIR / "06_features"
REPORT = OUTPUTS_DIR / "06_features"
CLEAR_WEEK = ("2022-06-12", "2022-06-19")  # local days, train, mostly clear
WINTER_WEEK = ("2022-12-12", "2022-12-19")


def load(horizon: int = 1) -> pd.DataFrame:
    return pd.read_parquet(STAGE / f"features_1H_D{horizon}.parquet")


def manifest() -> dict:
    return json.loads((STAGE / "features_manifest.json").read_text(encoding="utf-8"))


def k_distributions(t: pd.DataFrame) -> Result:
    """k with each candidate denominator on valid daytime train hours (D-11, T-13)."""
    m = t["daytime"] & (t["split"] == "train")
    rows, fig = [], plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    for den, colour in (("ghi", "#ef6c00"), ("poa", "#1565c0")):
        k = t.loc[m & t[f"k_valid_{den}"], f"k_{den}"]
        ax.hist(k, bins=np.linspace(0, 1.8, 91), histtype="step", lw=1.5, color=colour,
                label=f"k with {den.upper()}_cs")
        rows.append({"denominator": den, "hours": len(k), "median": round(float(k.median()), 4),
                     "p99": round(float(k.quantile(0.99)), 4),
                     "share_above_1": round(float((k > 1).mean()), 4),
                     "share_above_1.2": round(float((k > 1.2).mean()), 4),
                     "share_exactly_1.2": float((k == 1.2).mean())})
    for x in (1.0, 1.2):
        ax.axvline(x, color="0.5", ls=":", lw=1)
    ax.set_xlabel("hourly k = AC / theoretical clear-sky power")
    ax.set_ylabel("valid daytime train hours")
    ax.set_title("k with the two candidate denominators (train, daytime); no clipping",
                 fontsize=10)
    ax.legend(fontsize=8)
    table = pd.DataFrame(rows)
    t2 = table.set_index("denominator")
    return Result("s6_k_distributions", "k with each denominator", table=table, figure=fig,
                  facts={"share_above_1.2": t2["share_above_1.2"].to_dict(),
                         "share_above_1": t2["share_above_1"].to_dict(),
                         "median": t2["median"].to_dict()})


def weeks(t: pd.DataFrame) -> Result:
    """AC against the two theoretical powers in a clear summer and a winter week."""
    local = t.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharey=True)
    for ax, (a, b) in zip(axes, (CLEAR_WEEK, WINTER_WEEK), strict=True):
        w = t[(local >= pd.Timestamp(a)) & (local < pd.Timestamp(b))]
        x = w.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None)
        ax.plot(x, w["site_total_AC_kW"], color="k", lw=1.2, label="AC (measured)")
        ax.plot(x, w["theo_poa_kw"], color="#1565c0", lw=1, label="theo, POA_cs (panel 21/185)")
        ax.plot(x, w["theo_ghi_kw"], color="#ef6c00", lw=1, ls="--", label="theo, GHI_cs")
        ax.set_ylabel("kW")
        ax.set_title(f"{a} .. {b} (local)", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle("Why POA: in winter GHI_cs underestimates what a tilted panel receives, so "
                 "k with GHI rises above 1", fontsize=10)
    fig.tight_layout()
    return Result("s6_weeks", "Summer and winter week", figure=fig)


def imputation(m: dict) -> Result:
    """Share of imputed values per past-day feature, D+1."""
    d = pd.read_csv(REPORT / "feature_dictionary.csv")
    d = d[d["imputed_share_D1"] > 0].sort_values("imputed_share_D1", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(d["feature"], d["imputed_share_D1"] * 100, color="#455a64")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("% of hours imputed (D+1)")
    ax.set_title("Past-day features: gaps filled with the same-hour 30-day median", fontsize=10)
    table = d[["feature", "imputed_share_D1", "imputed_share_D2", "imputed_share_D3"]]
    return Result("s6_imputation", "Imputed shares", table=table, figure=fig, facts={
        "max_share_D1": float(d["imputed_share_D1"].max()),
        "masked_hours": m["masked_power_hours"], "warmup_hours": m["warmup_hours"],
        "dst_guard": m["dst_guard_removed_values"]})


def target_correlations(t: pd.DataFrame, horizon: int = 1, top: int = 20) -> Result:
    """Spearman correlation of each D+h feature with target_k on valid daytime train hours,
    after the warm-up: a first look before feature selection (stage 7)."""
    m = t["daytime"] & (t["split"] == "train") & t["k_valid"] & ~t["warmup"] \
        & ~t["op_limited_flag"]
    feats = features_for_horizon(horizon)
    corr = t.loc[m, feats].corrwith(t.loc[m, "target_k"], method="spearman")
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    show = corr.head(top)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(show.index, show.to_numpy(), color=np.where(show > 0, "#1565c0", "#c62828"))
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Spearman correlation with target_k")
    ax.set_title(f"Top {top} D+{horizon} features by |correlation| with k (train, daytime, "
                 "normal operation)", fontsize=9)
    table = pd.DataFrame({"feature": corr.index, "spearman_r": corr.round(4).to_numpy()})
    return Result("s6_target_correlations", "Feature correlations", table=table, figure=fig,
                  facts={"top": {k: round(float(v), 3) for k, v in show.head(5).items()},
                         "k_lag_1d_power": round(float(corr.get("site_total_AC_kW_lag_1d",
                                                                 np.nan)), 3)})


def kw_from_k(t: pd.DataFrame) -> dict:
    """Checks of the k <-> kW conversion on the real data (T-11)."""
    v = t["k_valid"]
    err = (t.loc[v, "target_k"] * t.loc[v, "theo_kw"] - t.loc[v, "site_total_AC_kW"]).abs()
    return {"max_abs_error_kw": float(err.max()), "valid_hours": int(v.sum()),
            "capacity_kw": AC_CAPACITY_KW}
