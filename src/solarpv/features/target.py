"""Stage 6: targets (PROJECT_MASTER.md §6.5, fix for finding F-06; docs/stage6_plan.md S3).

    theo_t = mean clear-sky irradiance over hour t [W/m2] * 3449 / 1000           [kW]
             (GHI_cs or POA_cs on the panel plane, decision D-11)
    k_t    = AC_t / theo_t        if theo_t > 50 kW            (ratio of hourly means)
    k_t    = 0, k_valid = False   otherwise
The original computed k per 15 minutes, clipped it at 1.2 and averaged the ratios, so k*theo did
not give back AC (F-06). Here k is a ratio of hourly means and is never clipped, so
AC_t = k_t * theo_t holds exactly (test T-11); k > 1.5 is flagged as suspect (K_SUSPECT_MAX),
never cut. A forecast k_hat >= 0 converts to kW as clip(k_hat * theo, 0, capacity).

theo is the mean over all 60 minutes of the hour, not only over the minutes with valid AC
(deviation from §6.4, docs/deviations.md): the theoretical power of a future hour must be
computable at issuance, and the same theo must invert k exactly.
"""

import numpy as np
import pandas as pd

from solarpv.config.constants import AC_CAPACITY_KW, K_SUSPECT_MAX, THEO_MIN_KW

DENOMINATORS = {"ghi": "ghi_clearsky", "poa": "poa_global_clearsky"}


def theoretical_power(clear_sky_irradiance: pd.Series) -> pd.Series:
    """Clear-sky power [kW] of the 3,449 kW plant for an hourly mean irradiance [W/m2]."""
    return clear_sky_irradiance * AC_CAPACITY_KW / 1000.0


def clear_sky_index(ac: pd.Series, theo: pd.Series, min_theo: float = THEO_MIN_KW
                    ) -> tuple[pd.Series, pd.Series]:
    """(k, k_valid). k is AC / theo where theo > min_theo and AC is known; 0 where theo is too
    small (night, deep twilight); NaN where AC is missing. Never clipped."""
    valid = (theo > min_theo) & ac.notna()
    k = pd.Series(0.0, index=ac.index)
    k[valid] = ac[valid] / theo[valid]
    k[ac.isna()] = np.nan
    return k, valid


def to_kw(k: pd.Series, theo: pd.Series, clip: bool = True) -> pd.Series:
    """kW from k. For forecasts (clip=True): clip(k * theo, 0, capacity). For the truth
    (clip=False) the exact inverse of clear_sky_index on valid hours."""
    kw = k * theo
    return kw.clip(0, AC_CAPACITY_KW) if clip else kw


def suspect(k: pd.Series) -> pd.Series:
    """k above K_SUSPECT_MAX (1.5): flagged for inspection, never removed or clipped."""
    return k > K_SUSPECT_MAX


def mean_daytime_k(k: pd.Series, valid: pd.Series, daytime: np.ndarray,
                   split: np.ndarray) -> float:
    """MEAN_DAYTIME_K: mean k over the valid daytime hours of train (evaluation mask F-02).
    It is the nRMSE denominator of the indirect target (§6.9)."""
    m = valid.to_numpy() & daytime & (split == "train")
    return float(k[m].mean())


def share_above(k: pd.Series, mask: np.ndarray, thresholds=(1.0, 1.2, 1.5)) -> dict:
    """Share of the hours in `mask` with k above each threshold (D-11 criterion 1, T-13)."""
    x = k[mask]
    return {f"k>{t}": round(float((x > t).mean()), 5) for t in thresholds}


def clear_days(irradiance: pd.Series, poa_cs: pd.Series, local_day, per_month: int = 6,
               min_elevation_mask=None) -> list:
    """Clear days chosen from the measured in-plane irradiance alone (smooth, high energy),
    independent of the denominator being judged. Hours used: min_elevation_mask (or all)."""
    m = np.ones(len(irradiance), bool) if min_elevation_mask is None else min_elevation_mask
    frame = pd.DataFrame({"day": local_day[m], "irr": irradiance.to_numpy()[m],
                          "cs": poa_cs.to_numpy()[m]})
    rows = []
    for d, g in frame.groupby("day"):
        p = g["irr"].to_numpy()
        if len(p) < 5 or np.isnan(p).any() or p.max() <= 0:
            continue
        rough = np.abs(np.diff(p, 2)).mean() / p.max()
        rows.append((d, p.sum() / max(g["cs"].sum(), 1e-9), rough))
    s = pd.DataFrame(rows, columns=["day", "kt", "rough"])
    s["month"] = [pd.Timestamp(d).strftime("%Y-%m") for d in s["day"]]
    keep = []
    for _, g in s.groupby("month"):
        g = g[g["kt"] >= g["kt"].quantile(0.6)]
        keep += list(g.nsmallest(per_month, "rough")["day"])
    return keep
