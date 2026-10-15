# Stage 5 test plan — cleaning

Written before the implementation. Every rule of `configs/cleaning_policy.yaml` is tested twice:
on synthetic data where the right answer is known (unit), and on the real output, where every
change must be explained by exactly one rule (contract). IDs continue PROJECT_MASTER.md §8.2.

## What "cleaned according to the format" means

1. **Every changed value is explained.** For each cleaned column, a minute differs from stage 4
   if and only if `rule_<column>` is non-zero, and the new value is what that rule prescribes
   (NaN for R1/R3/R5/R6, 0 for R2/R4).
2. **Every rule hit is justified.** A minute carries rule *r* only if the stage-4 value meets
   the condition of *r*, with the thresholds of the policy file.
3. **Flags never change values** (D-22): F1 (inverter stop) and F2 (curtailment) only mark.
4. **Real production is never removed** (T-16): no daytime value becomes 0, and no daytime
   value becomes NaN except as a range violation, its margin, a frozen run, or a dead DC
   sensor.
5. **The log tells the truth:** counts in `cleaning_log.csv` and the manifest equal the counts
   recomputed from the output; the manifest carries the hash of the policy file that was used.

## Tests

| ID | Level | Test | Pass criterion |
|---|---|---|---|
| T-22a | unit | R1 range: fault code, spike, sentinel | outside range → NaN; at the limit kept; only the bad minute and its margin change |
| T-22b | unit | R1 margin by source | ±1 min on the minute source, ±15 min on the polled source |
| T-22c | unit | R2 offset | −7.8 → 0; −25 → NaN (R1, not R2); positive values untouched |
| T-22d | unit | R3 frozen | 30 identical non-zero daytime minutes → NaN; 29 kept; a frozen run only at night kept; a run of exact zeros (inverter stop) kept; ambient needs 60 |
| T-22e | unit | R4 night | sun < 0 and \|AC\| ≤ 5 → 0; twilight 20 kW at −2° kept; daytime small value kept (T-16) |
| T-22f | unit | R5 stale night value | 1,296 kW at −30° → NaN |
| T-22g | unit | F1 downtime | sun up, irradiance 800, AC 0 → flagged, value unchanged; irradiance missing → clear-sky fallback; AC missing → not flagged |
| T-22h | unit | F2 curtailment | setpoint 50 in daytime → flagged; setpoint 100 → not; at night → not |
| T-22k | unit | R6 dead DC sensor | DC = 0 with AC 800 → NaN; DC = 0 with AC 50 kept; the AC column itself unchanged |
| T-22i | unit | rule precedence and idempotence | each minute gets the first applicable rule; cleaning twice = cleaning once |
| T-22j | unit | hourly flags | share of flagged minutes per hour; op-limited from 5 % (3 of 60 minutes) |
| T-23 | contract | **reconciliation**: changed ⇔ rule ≠ 0, new value matches the rule, rule condition holds on stage-4 values | 0 unexplained changes, 0 unjustified rule hits, every cleaned column |
| T-16 | contract | real production never removed | no daytime minute with stage-4 AC > 5 kW becomes 0; daytime AC NaNs only from R1/R3 |
| T-24 | contract | physical ranges after cleaning | no value outside the policy ranges; no temperature below −20; AC ≤ 1,897; no AC > 5 kW with the sun below −6° |
| T-25 | contract | flags never change values | on F1/F2 minutes without a value rule, cleaned = stage 4 |
| T-26 | contract | hourly output | 32,856 rows; values = `hourly_mean` of the cleaned minutes; `site_total_AC_kW` = inv1 + inv2; flag shares in [0, 1]; `op_limited_flag` = share ≥ 0.05; `hour_source` ∈ {minute, polled, mixed}, polled only from 2025-02-26 |
| T-27 | contract | energy on fault-free days | on days without R1/R3/R5 hits on AC, daytime AC energy after cleaning = stage 4 (±0.1 %) |
| T-28 | contract | log and manifest | per-rule counts recomputed from `rule_*` equal `cleaning_log.csv` and the manifest; policy hash matches `configs/cleaning_policy.yaml` |
| T-01 | leakage | quality columns can never be model inputs | every flag, count, rule, source column of the clean output is caught by `is_leaky` |

## Acceptance of stage 5 (PROJECT_MASTER.md §7)

All tests above green, plus: hourly maximum of `site_total_AC_kW` near 3,000 kW (spikes gone),
counts per rule recorded in PROJECT_MASTER.md and the report.
