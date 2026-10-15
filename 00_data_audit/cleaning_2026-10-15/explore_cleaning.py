"""Evidence for the stage-5 cleaning thresholds (PROJECT_MASTER.md §7 stage 5, F-14, F-17).

Runs on the clock-corrected minute data of stage 4 and prints, per topic, the distributions the
rules in configs/cleaning_policy.yaml are based on. Train period only where a threshold is
estimated from the data (2021-09 .. 2024-01), so no rule is tuned on val or test.
"""

import numpy as np
import pandas as pd

from solarpv.config.constants import TRAIN_END
from solarpv.config.paths import PROCESSED_DIR
from solarpv.config.site import load_site
from solarpv.data.clock import clear_sky_components

pd.set_option("display.width", 160)


def runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run id and run length for every element (equal consecutive values)."""
    changed = np.r_[True, values[1:] != values[:-1]]
    rid = np.cumsum(changed) - 1
    return rid, np.bincount(rid)[rid]


def main():
    df = pd.read_parquet(PROCESSED_DIR / "04_preprocess" / "scada_1min_true_utc.parquet")
    comp = clear_sky_components(df.index, load_site())
    elev = comp["elev"].to_numpy()
    train = df.index <= pd.Timestamp(TRAIN_END, tz="UTC")
    day = elev > 10

    print("== 1. temperature fault codes ==")
    for c in ("ambient_temp", "site_ambient_temp", "mod_temp1", "mod_temp2"):
        s = df[c]
        low = s[s < -20]
        lows = sorted(low.round(1).unique())[:6]
        print(f"{c:18s} min={s.min():8.2f} n<-20={len(low):6d} values={lows}  "
              f"p0.1={s[s > -20].quantile(0.001):.1f} p99.9={s.quantile(0.999):.1f} "
              f"max={s.max():.1f}")

    print("\n== 2. power maxima and spikes (per inverter) ==")
    for c in ("s01_inv1_AC_kW", "s01_inv2_AC_kW", "s01_inv1_DC_kW", "s01_inv2_DC_kW"):
        s = df[c]
        print(f"{c:16s} p99.99={s.quantile(0.9999):8.1f}  top5={s.nlargest(5).round(0).tolist()}")
    big = df[(df["s01_inv1_AC_kW"] > 1900) | (df["s01_inv2_AC_kW"] > 1900)]
    print(big[["s01_inv1_AC_kW", "s01_inv2_AC_kW", "s01_inv1_DC_kW", "s01_inv2_DC_kW"]].round(0))

    print("\n== 3. irradiance range ==")
    s = df["irradiance"]
    print(f"min={s.min():.1f} n<0={int((s < 0).sum())} n<-10={int((s < -10).sum())} "
          f"max={s.max():.1f} n>1400={int((s > 1400).sum())} n>1500={int((s > 1500).sum())}")
    print("negative irradiance by sun elevation:",
          pd.cut(pd.Series(elev[s.to_numpy() < 0]), [-90, -6, 0, 10, 90]).value_counts().to_dict())

    print("\n== 4. AC at night (sun below 0 deg), per inverter ==")
    night = elev < 0
    for c in ("s01_inv1_AC_kW", "s01_inv2_AC_kW"):
        s = df.loc[night, c]
        print(f"{c}: quantiles {s.quantile([0, .001, .01, .5, .99, .999, 1]).round(2).tolist()}")
    for lo, hi in ((-6, 0), (-90, -6)):
        m = (elev >= lo) & (elev < hi)
        s = df.loc[m, "s01_inv1_AC_kW"]
        print(f"  inv1 sun {lo}..{hi}: share |AC|>5 kW = {(s.abs() > 5).mean():.4f}, "
              f"max {s.max():.1f}")

    print("\n== 5. operativestate codes (train, sun > 10 deg) ==")
    for inv in ("inv1", "inv2"):
        st = df[f"s01_{inv}_state"].round()
        ac = df[f"s01_{inv}_AC_kW"]
        g = pd.DataFrame({"state": st[train & day], "ac": ac[train & day],
                          "irr": df.loc[train & day, "irradiance"]})
        tab = g.groupby("state").agg(n=("ac", "size"), ac_median=("ac", "median"),
                                     ac_p10=("ac", lambda x: x.quantile(.1)),
                                     irr_median=("irr", "median"))
        print(f"[{inv}]\n{tab[tab['n'] > 20].round(1)}")

    print("\n== 6. IEC mode and setpoint (sun > 10 deg) ==")
    d = df[day]
    ratio = d["site_total_AC_kW"] / (d["irradiance"].clip(lower=50) * 3.0)
    for name, key in (("iec_mode", d["iec_mode"].round()), ("iec_sp_w<100", d["iec_sp_w"] < 100)):
        t = pd.DataFrame({"key": key, "ratio": ratio, "ac": d["site_total_AC_kW"]})
        stats = t.groupby("key").agg(n=("ac", "size"), ac_median=("ac", "median"),
                                     ratio_median=("ratio", "median"))
        print(name, stats.round(3).to_dict("index"))
    sp = d["iec_sp_w"]
    print("setpoint values (daytime minutes):", sp.round().value_counts().head(10).to_dict())
    lim = d[sp < 100]
    print("when sp<100: AC / (sp% x 3449) quantiles:",
          (lim["site_total_AC_kW"] / (lim["iec_sp_w"].clip(lower=1) / 100 * 3449))
          .quantile([.5, .9, .99]).round(3).tolist())
    print("months with sp<100 (daytime minutes):",
          lim.groupby(lim.index.strftime("%Y-%m")).size().to_dict())
    m = d[d["iec_mode"].round() != 2]
    print("months with mode!=2 (daytime minutes):",
          m.groupby(m.index.strftime("%Y-%m")).size().to_dict())

    print("\n== 7. frozen daytime runs (sun > 10 deg), run length in minutes ==")
    for c in ("irradiance", "s01_inv1_AC_kW", "s01_inv2_AC_kW", "s01_inv1_DC_kW", "mod_temp1",
              "ambient_temp"):
        v = df[c].to_numpy()
        _, rl = runs(np.nan_to_num(v, nan=-9999))
        r = pd.Series(rl[day])
        counts = "  ".join(f">={k}: {(r >= k).sum():7d}" for k in (15, 30, 60, 120))
        print(f"{c:16s} daytime minutes in runs {counts}")
    v = df["s01_inv1_AC_kW"].to_numpy()
    rid, rl = runs(v)
    sel = day & (rl >= 30)
    if sel.any():
        frozen = pd.DataFrame({"run": rid[sel], "value": v[sel], "t": df.index[sel]})
        summary = frozen.groupby("run").agg(start=("t", "min"), minutes=("t", "size"),
                                            value=("value", "first"))
        print("inv1 AC daytime runs >= 30 min (first 15):\n", summary.head(15))
        print("values of those runs:", summary["value"].round(1).value_counts().head(8).to_dict())

    print("\n== 8. production missing while the sun shines (sun > 15 deg, irradiance > 300) ==")
    sunny = (elev > 15) & (df["irradiance"].to_numpy() > 300)
    for c in ("s01_inv1_AC_kW", "s01_inv2_AC_kW"):
        low = df[c].to_numpy() < 20
        print(f"{c}: sunny minutes with AC<20 kW: {int((sunny & low).sum())} "
              f"of {int(sunny.sum())}")
    both = sunny & (df["s01_inv1_AC_kW"].to_numpy() < 20) & (df["s01_inv2_AC_kW"].to_numpy() < 20)
    z = df[both]
    print("both inverters off while sunny, by month:",
          z.groupby(z.index.strftime("%Y-%m")).size().to_dict())
    print("their setpoint / mode / state1:", z["iec_sp_w"].round().value_counts().head(5).to_dict(),
          z["iec_mode"].round().value_counts().to_dict(),
          z["s01_inv1_state"].round().value_counts().head(5).to_dict())


def extra():
    """Sections 9-11: inverter balance, AC at deep night, frozen runs vs curtailment."""
    df = pd.read_parquet(PROCESSED_DIR / "04_preprocess" / "scada_1min_true_utc.parquet")
    elev = clear_sky_components(df.index, load_site())["elev"].to_numpy()
    train = df.index <= pd.Timestamp(TRAIN_END, tz="UTC")
    a1, a2 = df["s01_inv1_AC_kW"].to_numpy(), df["s01_inv2_AC_kW"].to_numpy()

    print("\n== 9. inverter balance AC1/AC2 (train, sun > 15, both > 200 kW) ==")
    m = train & (elev > 15) & (a1 > 200) & (a2 > 200)
    r = pd.Series(a1[m] / a2[m])
    print("quantiles:", r.quantile([.001, .01, .05, .5, .95, .99, .999]).round(3).tolist())
    m2 = train & (elev > 15) & (np.maximum(a1, a2) > 300)
    r2 = pd.Series(np.minimum(a1[m2], a2[m2]) / np.maximum(a1[m2], a2[m2]))
    print("min/max share below 0.9 / 0.7 / 0.5 / 0.1:",
          [round(float((r2 < t).mean()), 5) for t in (0.9, 0.7, 0.5, 0.1)])

    print("\n== 10. AC > 5 kW with the sun below -6 deg ==")
    for c in ("s01_inv1_AC_kW", "s01_inv2_AC_kW"):
        v = df[c].to_numpy()
        _, rl = runs(v)
        bad = (elev < -6) & (v > 5)
        print(f"{c}: {int(bad.sum())} minutes; inside constant runs >= 30 min: "
              f"{int((bad & (rl >= 30)).sum())}; days: "
              f"{sorted(set(df.index[bad].tz_convert('Asia/Jerusalem').strftime('%Y-%m-%d')))[:12]}")
    tw = (elev >= -6) & (elev < 0)
    v = df["s01_inv1_AC_kW"].to_numpy()
    _, rl = runs(v)
    print("twilight (-6..0) inv1 AC>5 not in long runs, by elevation band:",
          pd.cut(pd.Series(elev[tw & (v > 5) & (rl < 30)]), [-6, -3, -1, 0])
          .value_counts().to_dict(),
          "max:", float(v[tw & (rl < 30)].max()))

    print("\n== 11. frozen non-zero daytime runs vs setpoint ==")
    sp = df["iec_sp_w"].to_numpy()
    for c in ("s01_inv1_AC_kW", "irradiance"):
        v = df[c].to_numpy()
        _, rl = runs(v)
        fr = (elev > 10) & (rl >= 30) & (v != 0)
        print(f"{c}: frozen daytime minutes {int(fr.sum())}, of which setpoint<100: "
              f"{int((fr & (sp < 100)).sum())}, polled source: "
              f"{int((fr & (df['src_inv1' if 'AC' in c else 'src_meteo'].to_numpy() == 1)).sum())}")


if __name__ == "__main__":
    main()
    extra()
