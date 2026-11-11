"""Figures of stages 6 and 6b: the target, the features and the weather forecasts."""

import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle

from solarpv.config.constants import LAG_DROP
from solarpv.config.paths import REPO_ROOT
from solarpv.features.contract import LAG_DAYS, ROLLING_WINDOWS_DAYS
from solarpv.figures import _data as D
from solarpv.figures.registry import register
from solarpv.figures.style import C, blank, subplots

D11 = "outputs/06_features/d11_criteria.csv"
DERIVED = "configs/derived_constants.yaml"
DICTIONARY = "outputs/06_features/feature_dictionary.csv"
HORIZON_SKILL = "outputs/06b_gfs/horizon_skill.csv"
VALUE_CHECK = "outputs/06b_gfs/value_check.csv"
EXAMPLE_MONTH = "2022-03"


def _train_k():
    return D.table(D.FEATURES_D1, ["site_total_AC_kW", "theo_kw", "target_k", "k_valid",
                                   "daytime", "op_limited_flag", "sun_elevation", "split"],
                   split="train")


@register(
    "S06-01", "6", "target_construction", "From power to the clear-sky index on two days",
    "How is the indirect target built from the measured power?",
    "Top: measured AC power and the clear-sky power of the panel plane on a clear and a cloudy "
    "day of {month} (train). Bottom: their hourly ratio, the clear-sky index k, on daytime hours.",
    "On the clear day ({clear}) the measured power follows the clear-sky curve: the daily k is "
    "{clear_k:.2f}, with hourly values from {clear_min:.2f} to {clear_max:.2f} (the highest at "
    "low sun). On the cloudy day ({cloudy}) the daily k is {cloudy_k:.2f} and the hourly k drops "
    "to {cloudy_min:.2f}. Dividing by the clear-sky power removes the daily and seasonal shape "
    "and leaves the weather.",
    "k is the indirect target; it is converted back to kW by multiplying with the same "
    "clear-sky power, exactly, so both routes are scored in kW (F-06 fix, T-11).",
    (D.FEATURES_D1,))
def s06_target():
    f = _train_k()
    m = f.loc[EXAMPLE_MONTH].copy()
    m["day"] = D.local(m.index).date
    g = m.groupby("day").agg(ac=("site_total_AC_kW", "sum"), th=("theo_kw", "sum"),
                             lim=("op_limited_flag", "sum"))
    g = g[g["lim"] == 0]
    ratio = g["ac"] / g["th"]
    clear, cloudy = ratio.idxmax(), (ratio - 0.5).abs().idxmin()
    fig, axes = subplots(2, 2, height=4.2, sharex=True, sharey="row")
    facts = {"month": EXAMPLE_MONTH, "clear": str(clear), "cloudy": str(cloudy)}
    for j, (day, name) in enumerate(((clear, "clear"), (cloudy, "cloudy"))):
        d = m[m["day"] == day]
        hour = D.local(d.index).hour
        axes[0, j].plot(hour, d["theo_kw"], ls="--", color=C["grey"], label="clear-sky power")
        axes[0, j].plot(hour, d["site_total_AC_kW"], "o-", ms=3, color=C["blue"],
                        label="measured AC power")
        axes[0, j].set_title(f"{name} day, {day}")
        ok = d["daytime"] & d["k_valid"]
        axes[1, j].bar(hour[ok], d.loc[ok, "target_k"], color=C["orange"], width=0.7)
        axes[1, j].axhline(1, color="black", lw=0.8, ls=":")
        axes[1, j].set_xlabel("local hour")
        k = d.loc[ok, "target_k"]
        facts[f"{name}_k"] = float(d.loc[ok, "site_total_AC_kW"].sum() / d.loc[ok, "theo_kw"].sum())
        facts[f"{name}_min"] = float(k.min())
        facts[f"{name}_max"] = float(k.max())
    axes[0, 0].set_ylabel("kW")
    axes[1, 0].set_ylabel("clear-sky index k")
    axes[0, 0].legend(loc="upper left")
    return fig, facts


@register(
    "S06-02", "6", "clear_sky_reference", "Choosing the clear-sky reference of the target",
    "Should k be measured against horizontal clear-sky irradiance or against clear-sky "
    "irradiance on the panel plane?",
    "The three criteria fixed before the comparison, for the horizontal (GHI) and the "
    "panel-plane (POA) reference: share of train daytime hours with k above 1, spread of the "
    "monthly clear-day k, and validation error in kW of a fixed XGBoost (mean and SD of 3 seeds).",
    "The panel-plane reference wins all three: k above 1 in {poa_k1:.1%} of hours against "
    "{ghi_k1:.1%}, a monthly clear-day spread of {poa_spread:.3f} against {ghi_spread:.3f}, "
    "and {poa_nrmse:.2f} % against {ghi_nrmse:.2f} % of capacity on validation.",
    "k is defined against the panel-plane clear-sky power (D-11), with no clipping; the mean "
    "daytime k on train becomes the normalising constant of the indirect route.",
    (D11,))
def s06_reference():
    d = D.csv(D11).set_index("denominator")
    fig, axes = subplots(1, 3, height=2.6)
    specs = [("share_k_above_1", "share of hours with k > 1", 100, None),
             ("clear_day_k_spread", "spread of monthly clear-day k", 1, None),
             ("val_nrmse_kw_pct", "validation nRMSE (% of 3,449 kW)", 1, "val_nrmse_kw_seed_std")]
    for ax, (col, label, scale, err) in zip(axes, specs, strict=False):
        vals = [d.loc["ghi", col] * scale, d.loc["poa", col] * scale]
        errs = [d.loc["ghi", err], d.loc["poa", err]] if err else None
        bars = ax.bar(["horizontal\n(GHI)", "panel plane\n(POA)"], vals, yerr=errs, capsize=3,
                      color=[C["orange"], C["blue"]], width=0.6)
        bars[0].set_hatch("///")
        bars[0].set_edgecolor("white")
        ax.set_title(label, fontsize=9)
        for b, v in zip(bars, vals, strict=False):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8)
    axes[0].set_ylabel("lower is better")
    assert (d.loc["poa", [s[0] for s in specs]] < d.loc["ghi", [s[0] for s in specs]]).all()
    facts = {"poa_k1": float(d.loc["poa", "share_k_above_1"]),
             "ghi_k1": float(d.loc["ghi", "share_k_above_1"]),
             "poa_spread": float(d.loc["poa", "clear_day_k_spread"]),
             "ghi_spread": float(d.loc["ghi", "clear_day_k_spread"]),
             "poa_nrmse": float(d.loc["poa", "val_nrmse_kw_pct"]),
             "ghi_nrmse": float(d.loc["ghi", "val_nrmse_kw_pct"])}
    return fig, facts


@register(
    "S06-03", "6", "k_distribution", "The clear-sky index on train",
    "What values does the target take, and is any clipping needed?",
    "Histogram of hourly k on valid daytime train hours in bins of 0.025, with the mean daytime "
    "k used to normalise the indirect error.",
    "{n:,} valid daytime train hours, one peak near {mode:.2f}, mean daytime k {mean_k:.3f}. "
    "Only {gt12:.2%} of hours exceed 1.2, {gt12_low_sun:.0%} of them with the sun below 15 "
    "degrees, and no value sits at a cap, so k is kept as measured.",
    "No clipping: k converts back to the measured kW exactly, and the same rows feed the "
    "training masks of stage 9.",
    (D.FEATURES_D1, DERIVED))
def s06_k_hist():
    f = _train_k()
    rows = f[f["daytime"] & f["k_valid"] & f["target_k"].notna()]
    k = rows["target_k"]
    high = rows[rows["target_k"] > 1.2]
    mean_k = yaml.safe_load((REPO_ROOT / DERIVED).read_text(encoding="utf-8"))["mean_daytime_k"]
    fig, ax = subplots(height=2.7)
    bins = np.arange(0, 1.6 + 0.025, 0.025)
    counts, edges, _ = ax.hist(k.clip(upper=1.599), bins=bins, color=C["blue"], alpha=0.85)
    ax.axvline(mean_k, color=C["red"], ls="--", label=f"mean daytime k = {mean_k:.3f}")
    ax.axvline(1.2, color=C["grey"], ls=":", label="1.2")
    ax.set_xlabel("clear-sky index k (last bin includes values above 1.6)")
    ax.set_ylabel("train daytime hours")
    ax.legend(loc="upper left")
    facts = {"n": int(len(k)), "mode": float(edges[np.argmax(counts)] + 0.0125),
             "mean_k": float(mean_k), "gt12": float((k > 1.2).mean()),
             "gt12_low_sun": float((high["sun_elevation"] < 15).mean()),
             "at_12": int((k == 1.2).sum())}
    assert facts["at_12"] == 0
    return fig, facts


@register(
    "S06-04", "6", "history_per_horizon", "Which past days each horizon may use",
    "Which measured days are known when the forecast is issued at local midnight, for each "
    "horizon?",
    "One row per horizon; columns are local days relative to the issuance midnight (dashed "
    "line). Filled squares: the power and weather lags allowed at that horizon; bars below: the "
    "3- and 7-day same-hour windows.",
    "Lags allowed, in days back from the target day: D+1: {lags1}; D+2: {lags2}; D+3: {lags3}. "
    "The same-hour windows always end on the last day before issuance (day 0), whatever the "
    "horizon.",
    "Every history feature is built from days known at issuance, and a test plants a future "
    "day to check that it is caught (T-09, T-18).",
    ())
def s06_history():
    fig, ax = subplots(height=2.9)
    blank(ax)
    days = list(range(-13, 4))
    x0, cw = 0.12, 0.85 / len(days)
    facts = {}
    for i, d in enumerate(days):
        ax.text(x0 + (i + 0.5) * cw, 0.95, str(d), ha="center", fontsize=7.5, color=C["grey"])
    t0x = x0 + (days.index(0) + 1) * cw
    ax.plot([t0x, t0x], [0.05, 0.92], ls="--", color=C["red"], lw=1)
    ax.text(t0x, 0.0, "issuance, 00:00 local", ha="center", fontsize=7.5, color=C["red"])
    for r, h in enumerate((1, 2, 3)):
        y = 0.72 - r * 0.27
        ax.text(0.02, y + 0.04, f"D+{h}", fontsize=9, va="center")
        dropped = {int(s.split("_")[1].rstrip("d")) for s in LAG_DROP[h]}
        lags = [lag for lag in LAG_DAYS if lag not in dropped]
        facts[f"lags{h}"] = ", ".join(map(str, lags))
        for i, d in enumerate(days):
            xi = x0 + i * cw
            face = "white"
            if d == h:
                face = C["yellow"]
            elif (h - d) in lags:
                face = C["blue"]
            ax.add_patch(Rectangle((xi + 0.002, y), cw - 0.004, 0.08, fc=face, ec=C["grey"],
                                   lw=0.5))
        for j, n in enumerate(ROLLING_WINDOWS_DAYS):
            first, last = 0, -(n - 1)
            xa = x0 + days.index(last) * cw
            xb = x0 + (days.index(first) + 1) * cw
            ax.plot([xa + 0.003, xb - 0.003], [y - 0.03 - j * 0.035] * 2,
                    color=[C["orange"], C["green"]][j], lw=3, solid_capstyle="butt")
    ax.text(x0, -0.06, "yellow: target day   blue: lag days   orange / green bars: 3- and 7-day "
            "same-hour windows", fontsize=7.5)
    ax.set_ylim(-0.1, 1)
    return fig, facts


@register(
    "S06-05", "6", "feature_table", "The candidate feature table per horizon",
    "How many candidate features of each kind exist at each horizon, and how much had to be "
    "filled in?",
    "Stacked bars: candidate features per group at each horizon (contract only, before the "
    "weather forecasts are added).",
    "{t1} / {t2} / {t3} candidates at D+1 / D+2 / D+3; only the power and weather lags shrink "
    "with the horizon. At most {imp:.2%} of any feature's values are filled from the same hour "
    "of the previous 30 days.",
    "The table goes to feature selection together with the weather-forecast candidates of "
    "stage 6b.",
    (DICTIONARY,))
def s06_table():
    d = D.csv(DICTIONARY)
    groups = list(d["group"].value_counts().index)
    fig, ax = subplots(height=2.4)
    hatches = [None, "///", "...", "xx", "\\\\"]
    colours = [C["blue"], C["orange"], C["green"], C["purple"], C["sky"]]
    left = np.zeros(3)
    totals = []
    for gi, g in enumerate(groups):
        n = [int(((d["group"] == g) & d["horizons"].astype(str).str.contains(str(h))).sum())
             for h in (1, 2, 3)]
        ax.barh(["D+1", "D+2", "D+3"], n, left=left, color=colours[gi], hatch=hatches[gi],
                edgecolor="white", label=g.replace("_", " "))
        left += n
    totals = [int(v) for v in left]
    for i, v in enumerate(totals):
        ax.text(v + 0.8, i, str(v), va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("candidate features")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=5)
    imp = d[[c for c in d.columns if c.startswith("imputed_share")]].max().max()
    facts = {"t1": totals[0], "t2": totals[1], "t3": totals[2], "imp": float(imp)}
    return fig, facts


@register(
    "S6b-01", "6b", "forecast_availability", "Forecast runs public before the issuance time",
    "Was every weather-forecast run really available at local midnight, when the forecast is "
    "issued?",
    "Histogram of the time between the upload of the last file of the run used for each day "
    "and the issuance time; runs that replaced a late 12Z run are counted separately.",
    "{n_days:,} issuance days; the run used was uploaded {min_h:.1f} to {max_h:.1f} hours "
    "before issuance (median {med_h:.1f} h). {n_fallback} days use the earlier 06Z run, because "
    "their 12Z files had been uploaded again after issuance.",
    "Only runs public before issuance enter the features (T-17); the check reads the upload "
    "time of every file, not the nominal run time.",
    (D.GFS_D1,))
def s6b_availability():
    g = D.table(D.GFS_D1, ["t0_utc", "gfs_init", "gfs_fallback", "gfs_uploaded_max"])
    g = g.drop_duplicates("t0_utc")
    margin = (pd.to_datetime(g["t0_utc"], utc=True) -
              pd.to_datetime(g["gfs_uploaded_max"], utc=True)).dt.total_seconds() / 3600
    fb = g["gfs_fallback"].astype(bool).to_numpy()
    fig, ax = subplots(height=2.5)
    bins = np.arange(np.floor(margin.min()), np.ceil(margin.max()) + 0.25, 0.25)
    ax.hist([margin[~fb], margin[fb]], bins=bins, stacked=True, color=[C["blue"], C["orange"]],
            label=["12Z run", "earlier run (12Z re-uploaded late)"])
    ax.axvline(0, color=C["red"], ls="--", lw=1)
    ax.text(0, ax.get_ylim()[1] * 0.9, " issuance", color=C["red"], fontsize=8)
    ax.set_xlabel("hours from upload of the run to issuance")
    ax.set_ylabel("issuance days")
    ax.legend(loc="upper right")
    assert (margin > 0).all()
    facts = {"n_days": int(len(g)), "min_h": float(margin.min()), "max_h": float(margin.max()),
             "med_h": float(margin.median()), "n_fallback": int(fb.sum())}
    return fig, facts


@register(
    "S6b-02", "6b", "gfs_agreement", "Forecast agreement with the measured clear-sky index",
    "How well does the forecast cloudiness track what the plant measured, at each horizon?",
    "Correlation between the forecast clear-sky index on the panel plane and the measured k, "
    "per hour and per day, at each horizon (train and validation days).",
    "Daily correlation {d1:.2f} / {d2:.2f} / {d3:.2f} and hourly {h1:.2f} / {h2:.2f} / "
    "{h3:.2f} at D+1 / D+2 / D+3 over {days:,} days: a real, imperfect forecast that weakens "
    "with lead time.",
    "The forecast variables became candidates for feature selection, after the value test of "
    "S6b-03.",
    (HORIZON_SKILL,))
def s6b_agreement():
    s = D.csv(HORIZON_SKILL)
    fig, ax = subplots(height=2.5)
    x = np.arange(3)
    ax.bar(x - 0.18, s["corr_daily"], 0.34, color=C["blue"], label="daily")
    ax.bar(x + 0.18, s["corr_hourly"], 0.34, color=C["orange"], hatch="///", edgecolor="white",
           label="hourly")
    ax.set_xticks(x, ["D+1", "D+2", "D+3"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("correlation with measured k")
    ax.legend(loc="upper right", ncols=2)
    facts = {f"d{h}": float(s.loc[i, "corr_daily"]) for i, h in enumerate((1, 2, 3))}
    facts.update({f"h{h}": float(s.loc[i, "corr_hourly"]) for i, h in enumerate((1, 2, 3))})
    facts["days"] = int(s["days"].iloc[0])
    return fig, facts


@register(
    "S6b-03", "6b", "gfs_value", "Value of the weather forecasts on validation",
    "Do the forecast variables improve a fixed model enough to be kept?",
    "Validation nRMSE in kW of a fixed XGBoost with the contract features alone (dashed, "
    "circles) and with the forecast variables added (solid, squares); mean and SD over 3 seeds.",
    "Adding the forecasts lowers the error by {gain_min:.2f} to {gain_max:.2f} points of "
    "capacity at every horizon and in both routes, against a seed spread of at most "
    "{sd_max:.2f}.",
    "The forecasts were accepted (D-12) and entered feature selection as candidates.",
    (VALUE_CHECK,))
def s6b_value():
    v = D.csv(VALUE_CHECK)
    fig, axes = subplots(1, 2, height=2.6, sharey=True)
    gains = []
    for ax, target in zip(axes, ("indirect", "direct"), strict=False):
        t = v[v["target"] == target]
        for feats, style, col in (("contract", "o--", C["grey"]),
                                  ("contract+gfs", "s-", C["blue"])):
            r = t[t["features"] == feats].sort_values("horizon")
            ax.errorbar(r["horizon"], r["nrmse_kw_mean"], yerr=r["nrmse_kw_std"], fmt=style,
                        color=col, capsize=3, ms=4,
                        label="contract only" if feats == "contract" else "with forecasts")
        a = t[t["features"] == "contract"].set_index("horizon")["nrmse_kw_mean"]
        b = t[t["features"] == "contract+gfs"].set_index("horizon")["nrmse_kw_mean"]
        gains += list(a - b)
        ax.set_title(f"{target} route")
        ax.set_xticks([1, 2, 3], ["D+1", "D+2", "D+3"])
    axes[0].set_ylabel("validation nRMSE (% of 3,449 kW)")
    axes[0].legend(loc="lower left")
    facts = {"gain_min": float(min(gains)), "gain_max": float(max(gains)),
             "sd_max": float(v["nrmse_kw_std"].max())}
    return fig, facts
