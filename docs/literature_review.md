# Literature review: weather-forecast-based day-ahead PV forecasting

Requested by the user on 2026-10-20: an in-depth review of the most relevant recent papers, with
insights for every stage of the project and new ideas to implement. The ideas are registered as
**L-01 … L-20** (§4) and are linked from PROJECT_MASTER.md §18 and from each stage in §7. None of
them is a decision: each one is tested on **validation only**, at the stage named, and adopted
only if it helps there (D-12 logic). The test split is never used to choose between them.

## 0. How the review was done

- **Search.** Web and scholarly indexes (OpenAlex, Crossref, publisher and repository pages),
  restricted to: day-ahead PV power or irradiance forecasting driven by numerical weather
  prediction (NWP); machine learning and deep learning for that task; verification of solar
  forecasts; aerosol and dust effects on solar forecasts (the site is in the hyper-arid Arava);
  and evaluations of GFS, the NWP source chosen in D-12.
- **Reading.** Five papers were read in full from open-access copies (core, §1). Five more are
  used only at abstract or thesis level, and are marked so (supporting, §2). Paywalled full texts
  that could not be opened legally were not used, and nothing is claimed from them beyond their
  abstracts.
- **Citations.** Every reference was checked against its Crossref or OpenAlex record (title,
  authors, venue, volume, DOI).
- **Text.** Everything below is a paraphrase. Numbers are the papers' own.
- **Transfer caveat.** Most of the strongest work comes from one group (BME Budapest, with
  D. Yang), on Hungarian plants with ECMWF forecasts at 15-minute resolution. Our case differs
  in four ways:
  - the climate: hyper-arid, and 78–98 % of daytime hours are clear (stage 3b, Q27);
  - the NWP: GFS, which is coarser;
  - the resolution: hourly;
  - the horizons and the targets: D+1…D+3 from local midnight, predicting k and kW.
  Their numbers are therefore not targets for us; their *methods* are what transfers.

## 1. Core papers (read in full)

### C1. Markovics, Békési & Mayer (2026): deep learning vs classical machine learning on NWP

*A unified benchmark of deep learning and classical machine learning models for weather-based
solar power forecasting: Balancing complexity and skill.* Energy and AI 25, 100833.
doi:10.1016/j.egyai.2026.100833 (open access, CC BY).

- **Setting:**
  - Plants: 7 Hungarian PV plants (0.59–20 MW DC).
  - Data: 15-minute data from 2020–2023, daytime only.
  - NWP: ECMWF HRES (9 km, 00 UTC run, nearest grid point), with GHI, 2-m temperature and
    10-m wind. GHI was interpolated to 15 minutes through the clear-sky index; the others
    linearly.
  - Split: train 2020–2021, validation 2022, test 2023.
- **Models (12):**
  - Classical: kernel ridge (KR), Extra Trees (ET), XGBoost, LightGBM, MLP.
  - Deep: CNN, RNN (LSTM, GRU, bidirectional), Transformer, Autoformer, Informer, Pyraformer,
    FEDformer.
  - Tuning: Optuna (TPE), 300 trials per model and site.
- **Predictor sets:**
  - *basic*: NWP plus solar position;
  - *hybrid*: basic plus outputs of a physical model chain (clear-sky PV power, beam, diffuse
    and tilted irradiance, cell temperature, PV power);
  - *feature engineering*: daily means and centred 1-, 3- and 5-hour moving averages, with
    near-duplicates removed (|r| > 0.98).
- **Verification:**
  - nRMSE normalised by mean daytime power;
  - skill against 2-day persistence;
  - Diebold–Mariano tests with Newey–West (lag 48);
  - greedy equal-weight blending.
- **Results (test year, mean over plants):**
  - 2-day persistence: nRMSE 81.6 %.
  - Physical model chain: 48.7 % (skill 40.3 %).
  - The tuned models lie between 42.7 % (RNN, Pyraformer) and 43.5 % (XGBoost, LightGBM):
    - ET with hybrid and engineered features: 43.2 %, trained in about 3 s;
    - transformers: about 400–1,000 s per trial, often more.
  - Blending the best complementary models (KR + RNN, …) gains about 0.8 percentage points, as
    much as the spread between the best and the worst model.
  - Gradient boosting and transformers were the most sensitive to the tuning budget; KR and RNNs
    the least.
  - Physical hybrid features helped the tree ensembles. Temporal feature engineering is
    essential for tabular models, while sequence models need it less.
- **Authors' recommendation:**
  - a physically hybridised ET as the cheap baseline;
  - an RNN for the best single model;
  - blending when affordable;
  - large transformers only with a large budget.

**Why it matters here:** this is the closest benchmark to our stage 9/10 design (XGB, BiLSTM,
PatchTST on NWP). With good predictors and fair tuning, the architecture moves the error by
less than 1 point. **Predictor construction, tuning protocol and blending matter more than the
choice of network.**

### C2. Mayer, Yang & Markovics (2025): from ensemble NWP to a deterministic PV forecast

*The complexity and dimensionality of making deterministic photovoltaic power forecasts from
ensemble numerical weather prediction.* Energy Conversion and Management 344, 120303.
doi:10.1016/j.enconman.2025.120303 (open access).

- **Setting:**
  - Plants: 5 Hungarian plants (0.5–20 MW), data 2019–2022.
  - NWP: ECMWF HRES (deterministic) and ENS (50 perturbed members), 00 UTC run, with GHI,
    2-m temperature and 10-m wind.
  - Split: fitting on 2019–2021, verification on 2022.
- **Workflows:** 16, from four yes/no choices:
  1. convert each member to power, or average first;
  2. calibrate the ensemble (linear quantile regression) or not;
  3. post-process between the model chain and the mean, or not;
  4. bias-correct the final power (kernel conditional density estimation on cos zenith,
     predicted clear-sky index and temperature) or not.
- **Physical chain:** Yang separation, Perez transposition, Martin–Ruiz reflection, Mattei cell
  temperature, Beyer module model, constant-efficiency inverter with clipping.
- **End-to-end ML:** ET, LightGBM, XGBoost and MLP, fed all 50 members (Optuna, 500 iterations).
- **Results (RMSE normalised by mean power):**
  - Bias correction of the *final power* alone gives −4.9 % against raw HRES with the chain.
  - The best workflow (members to power, calibrate, average, bias-correct) gives −10.1 %
    (nRMSE 37.4 %).
  - Every workflow that ends with the power bias correction lands within 37.4–37.6 %.
  - Ensembles beat the deterministic forecast by about 5 %.
  - Post-processing the irradiance adds little if the goal is power.
  - The MLP approaches the best chain. The tree models lag (collinear member inputs).
  - ML beats the chain around midday but loses early and late in the day, where it has fewer
    samples.
  - ML needs about 2–3 years of training data.
- **Authors' recommendations:**
  - the bias correction of the final power is the only indispensable post-processing step;
  - physical chains stay valuable for interpretability;
  - combine ML and physical forecasts, whose errors differ.

**Why it matters here:**
- It defines *where* post-processing belongs: at the end, on power or k.
- It suggests a cheap physical baseline and hybrid features.
- It warns that ML is weakest at low sun — the same hours where stage 3b found the clear-sky
  index unstable (Q22).

### C3. Mayer & Yang (2024): which skill score, and fair comparison across loss functions

*Potential root mean square error skill score.* Journal of Renewable and Sustainable Energy 16,
016501. doi:10.1063/5.0187044 (open access preprint: arXiv:2409.14263).

- **Skill against CLIPER:** the RMSE skill score of Yang et al. (2020, S3) uses as reference the
  MSE-optimal convex combination of climatology and persistence (CLIPER). Its error has a closed
  form, RMSE_cp = √(1 − γ²(h)) · σ, where γ(h) is the lag-h autocorrelation of the observations
  and σ their standard deviation. The skill is then s = 1 − RMSE / RMSE_cp.
- **The problem:** this skill is unfair to forecasts optimised for another loss (for example
  MAE or a quantile), because they are not calibrated for MSE.
- **The proposal:** the *potential* skill score, s = 1 − √(1 − ρ²) / √(1 − γ²(h)), where ρ is
  the correlation between forecast and observation. It is the skill after an MSE-optimal linear
  recalibration, and needs no actual recalibration.
- **Further guidance:**
  - judge a forecast with the metric it was trained for;
  - skill scores of different metrics are not comparable.
- **Case study:** 32,400 physical chains on Hungarian plants. The MSE-optimised forecast has an
  actual / potential RMSE skill of 36.1 / 36.4 %; the MAE-optimised one 33.7 / 36.1 %.

**Why it matters here:**
- Our skill reference today is persistence only (§6.9). CLIPER is the community standard and is
  much harder to beat at a clear-sky desert site, where persistence is already good.
- Our models use different losses (XGB: MSE; BiLSTM: quantile loss), so the potential skill
  keeps the comparison fair.

### C4. Hou, Papachristopoulou & Kazadzis (2025): aerosol forecasts for day-ahead clear-sky irradiance

*Assessment of aerosol optical depth forecast for day-ahead clear-sky direct irradiance.*
Atmospheric Measurement Techniques 18, 4543–4557. doi:10.5194/amt-18-4543-2025 (open access,
CC BY).

- **Setting:**
  - 21 AERONET sites (2010–2020), among them the desert-dust sites Kuwait, Tamanrasset and
    Banizoumbou. There is no site in Israel.
  - Three day-ahead AOD sources: persistence of yesterday's AERONET AOD, an AERONET monthly
    climatology, and the CAMS forecast (0.4°, hourly, up to 5 days, available since 2015).
  - Clear-sky direct irradiance computed with libRadtran.
- **Results:**
  - CAMS is the best AOD source at the most sites (10 of 21).
  - At dust sites the result is mixed, and the abstract favours persistence. At Kuwait, 50–60 %
    of days had a DNI error above 20 % with CAMS, against under 40 % with AERONET-based methods.
  - No source keeps the day-ahead clear-sky DNI error under 5 %; 20 % is achievable at most
    sites.
  - The day-to-day variability of aerosols is what drives the error.
  - Daily mean CAMS AOD worked better than hourly.

**Why it matters here:**
- Clouds are rare at Ketura, so aerosols (dust, especially the spring khamsin) are a main
  remaining source of clear-sky error.
- GFS radiation does not forecast dust (see C5).
- Two aerosol predictors are worth testing: a persistence-type one, which our own measurements
  can provide at no cost, and the CAMS forecast.

### C5. Zhang, Shen, Xia & Shi (2020): how good is raw GFS irradiance, day ahead?

*Validation of GFS day-ahead solar irradiance forecasts in China.* Preprint, arXiv:2007.01639
(not peer-reviewed as far as the indexes show).

- **Setting:**
  - GFS 0.5° product from the 18 UTC cycle, at 3-hourly steps. DSWRF was interpolated to hourly
    values through the clearness index.
  - 17 national radiation stations and one BSRN station (2014–2016), quality-controlled, with
    solar elevation above 5°.
- **Results:**
  - Raw GFS GHI is biased high at most stations; the authors attribute this partly to aerosols
    under-represented in GFS's climatology.
  - Raw GFS was worse than simple persistence at every station.
  - An XGBoost model on GFS variables plus sun geometry cut the error at the BSRN site (GHI rRMSE
    38.2 %, against 79.6 % for raw GFS and 50.1 % for persistence). Its inputs: clearness index,
    pressure, temperature, relative humidity, sunshine duration, precipitable water and total
    cloud cover. Predictors correlated above 0.9 were dropped.
  - Recommendations: MOS-type correction and comparison with other NWP models.

**Why it matters here:**
- It is the only full-text evaluation we found of the *same NWP product family* we will use.
- Raw GFS must not be used as is; it is worth something only through post-processing or as ML
  predictors.
- Its candidate variables (precipitable water, sunshine duration, cloud cover) extend our list
  in §6.12.
- Its interpolation through the clearness index is the right way to go from 3-hourly to hourly
  values.

## 2. Supporting papers (abstract or thesis level)

| ID | Reference | What it adds (from the abstract or thesis, not the full paper) |
|---|---|---|
| S1 | Markovics & Mayer (2022). *Comparison of machine learning methods for photovoltaic power forecasting based on numerical weather prediction.* Renewable and Sustainable Energy Reviews 161, 112364. doi:10.1016/j.rser.2022.112364 | 24 ML models, 16 Hungarian plants, day ahead. Kernel ridge and MLP were best (up to 44.6 % skill over persistence). **Adding sun-position angles and statistically processed irradiance cut RMSE by 13.1 %.** Hyperparameter tuning is essential for the less robust models |
| S2 | Mayer & Gróf (2021). *Extensive comparison of physical models for photovoltaic power forecasting.* Applied Energy 283, 116239. doi:10.1016/j.apenergy.2020.116239; and M. J. Mayer's PhD thesis booklet (BME, 2020) | All combinations of separation, transposition, reflection, cell-temperature, module, shading and inverter models on 16 plants. Transposition and separation matter most, the inverter least. The best and worst chains differ by about 12–13 % in RMSE and MAE. **MAE and RMSE conflict:** RMSE-optimal chains are simple and capture only 78–85 % of the true variance (under-dispersion). Forecast wind speed adds nearly nothing: a constant mean wind did as well |
| S3 | Yang, D., et al. (2020). *Verification of deterministic solar forecasts.* Solar Energy 210, 20–37. doi:10.1016/j.solener.2020.04.019 | Community guideline (33 authors): the Murphy–Winkler distribution-oriented framework (calibration, resolution, conditional biases) and an RMSE skill score against the optimal climatology–persistence combination (details through C3) |
| S4 | Varga, G., Gresina, F., Gelencsér, A., Csávics, A., & Rostási, Á. (2026). *The shadow of the wind: the impact of Saharan dust on photovoltaic power generation in the Mediterranean.* Renewable Energy 256, 124337. doi:10.1016/j.renene.2025.124337 | 2019–2023, five Mediterranean countries. Saharan dust events cut PV output by 25–40 % on average, and by more than 50 % in extreme cases. Day-ahead forecasts erred by up to −15 % (Iberia) and +10 % (Italy, Greece). Calls for dust-aware forecasting |
| S5 | Mathiesen, P., & Kleissl, J. (2011). *Evaluation of numerical weather prediction for solar irradiance forecasting in the CONUS* (AMS Annual Meeting, abstract 180986) | GFS, NAM and ECMWF against ground data in the US. The NAM over-forecast clear skies and under-forecast overcast ones; a MOS correction cut its monthly bias from up to 25 % to within ±5 %. Conference abstract only |

## 3. What the literature agrees on (synthesis)

1. **Day ahead, the information is in the NWP.** Persistence of the sky condition is weak beyond
   a day; our own data agree (stage 3b, Q26: clear-sky-index autocorrelation 0.34 at one day,
   0.12 at fourteen). Past-day lag features will add little beyond the clear-sky features (C1,
   C5, S1).
2. **Predictors and tuning beat architecture.** The spread between well-tuned models is under
   one point of nRMSE (C1). The large gains come from:
   - sun position and processed irradiance (−13.1 % RMSE, S1);
   - physical hybrid features (C1);
   - fair tuning (C1, S1).
3. **Raw NWP irradiance is biased, and must be post-processed or learned from.** This holds
   especially for GFS (C5, S5). The single post-processing step that always pays is the bias
   correction of the *final* power forecast (C2).
4. **Physical knowledge helps machine learning.** It helps through hybrid features (C1), through
   clear-sky normalisation (all papers interpolate or normalise through the clear-sky index), and
   through a physical baseline (C1, C2, S2).
5. **Verification needs a hard reference and consistent metrics.** Use CLIPER skill (S3, C3),
   judge with the metric the model was trained for, and remember that MSE-optimal forecasts are
   under-dispersed (S2, C2: variance ratio ≈ ρ²).
6. **Where the sky is clear, aerosols matter.** Day-ahead aerosol forecasts are imperfect, but
   dust events are large and can be partly anticipated (C4, S4). GFS does not forecast dust.
7. **Deep learning is not a free win.** RNNs were the best single models, transformers were not
   better and cost much more, and blending helps (C1). ML is weaker at low sun (C2).

## 4. Insights and ideas by project stage

Status of every idea: ⬜ proposed. Each is decided at its stage, **on validation only**, and the
decision is recorded in PROJECT_MASTER.md (D-11, D-12, D-13, D-08 or a new D-xx). Priority:
H high (cheap and likely useful), M medium, L low.

### Stages 2–5 and 3b (done): what the literature confirms

- **Clear-sky normalisation and true solar geometry are prerequisites** of every method reviewed.
  That is exactly what F-01 (time base) and D-19 (clocks) fixed.
- **Quality control before training** (C5: limits, closure tests, sun above 5°; C1: daytime
  only, maintenance zeros removed) matches stage 5. The literature removes daytime zeros as
  maintenance; we *flag* them instead (D-22), which is stricter and keeps the evaluation honest.
- **Partial outages** (our F-22) are not discussed by any of the papers. They clean data from
  operator records, so flag F3 is a contribution of this project.

### Stage 6: target and features

| ID | Idea | Source | Pri. |
|---|---|---|---|
| L-01 | **Physical model-chain features and baseline from GFS.** GFS GHI → separation (Erbs or Engerer) → Perez transposition to the measured panel plane (21°/185°, F-23) → cell temperature (Faiman, from GFS 2-m temperature; wind constant, S2) → PV power (PVWatts, with γ = −0.42 %/K measured in stage 3). Produces `fcst_gfs_poa`, `fcst_gfs_tcell` and `fcst_gfs_pv_phys`, used (a) as hybrid predictors and (b) as a physical baseline model | C1, C2, S2 | H |
| L-02 | **Temporal engineering of the NWP features:** for the target day, the daily mean and centred 1-, 3- and 5-hour moving averages of GFS kt, GHI and cloud cover; drop near-duplicates (\|r\| > 0.98) | C1 | H |
| L-03 | **NWP clear-sky index as a direct predictor of k:** `fcst_gfs_kt = GHI_gfs / GHI_cs` (and the POA version from L-01), aligned with the target definition of D-11. The GFS bias is then learned in k space, where it is most stable | C1, C5 | H |
| L-04 | **Low sun:** report the error by sun-elevation band. Test on validation a sample weight proportional to the clear-sky power, or a higher validity threshold than theo > 50 kW for the k loss (stage 3b, Q22: k above 1.2 at 03 and 15 UTC) | C2, Q22 | M |

**Status after stage 6 (2026-10-20):**
- **L-04 tested, not adopted.** k is least stable at low sun (p99 1.42 at 5–10°), but leaving hours with theo below 100 or 150 kW out of the training loss gave no gain beyond the seed noise: 0.053 pp, against 0.119 pp needed (`docs/run_journal.md`, R6-04).
- **L-01 to L-03 need the GFS data** and move to stage 6b.

### Stage 6b: GFS (D-12, §6.12)

| ID | Idea | Source | Pri. |
|---|---|---|---|
| L-05 | **Use the 0.25° GFS files with hourly steps** (`pgrb2.0p25`, leads 9–82 h are within the hourly range) instead of the 1° 3-hourly files: no temporal interpolation of irradiance, and a closer grid point. To verify at the start of 6b: hourly availability and the AWS archive coverage for Aug 2021 – May 2025. If only 3-hourly data are used, interpolate through the clear-sky index, not linearly | C1, C2, C5 | H |
| L-06 | **De-averaging DSWRF correctly:** the GFS radiation fields are averages from the start of each 6-hour window. Hourly mean over (t−1, t] = t·A(t) − (t−1)·A(t−1) within a window, restarting at each window boundary (read the start hour from the GRIB step range, or use `wgrib2 -ncep_norm`). Verify on a sample file first, and add a synthetic unit test (stage 6b) | GFS documentation; C5 | H |
| L-07 | **More GFS candidates:** low, middle and high cloud cover, precipitable water, relative humidity, sunshine duration (SUNSD), 2-m temperature. Wind gets low priority (S2: marginal) | C5, S2 | M |
| L-08 | **Spatial context:** mean and standard deviation of cloud cover and kt over the 3×3 neighbourhood, as variability and uncertainty predictors | C1/C2 use the nearest point only; spatial refinement helped in studies not read in full | L |
| L-09 | **Final bias correction (MOS) of the forecast k or power:** a model-agnostic last layer. Linear, or kernel-weighted on cos zenith, GFS kt and temperature. Fitted on out-of-fold train predictions and chosen on validation. It may be redundant for XGB (ML learns the bias, C2), so it is tested, not assumed | C2, C5, S5 | M |
| L-10 | **GEFS ensemble** (31 members, AWS `noaa-gefs-pds`): ensemble mean and spread of GHI and cloud cover. Convert members to k or power first, then average. About 5 % gain in C2, at roughly 31× the download. Only after L-05 works | C2 | L–M |
| L-11 | **Dust and aerosol predictors:** (a) **yesterday's clear-hour deficit** at the site: the measured in-plane irradiance divided by clear-sky POA, on the clear hours of day D−1. It is known at T0, costs nothing, and is the analogue of AERONET persistence in C4; (b) the **CAMS** day-ahead AOD (550 nm) and dust AOD, as a daily mean (better than hourly in C4). This needs a free ADS account, which the user must create; (c) GEFS-Aerosols AOD, if an archive exists. Leakage rule as T-17 (issue time + latency ≤ T0) | C4, S4 | H (a), M (b) |

**Status after stage 6b (2026-10-25):**
- **L-05:** checked. The hourly 0.25° files cost about ten times the 1° transfer, so 1° is
  used, with hourly values through the clear-sky index.
- **L-06:** implemented and tested. SUNSD turned out to be accumulated in the same windows.
- **L-01, L-02, L-03, L-07, L-08 and L-11a:** implemented as candidates. As a group they
  were admitted by the value check (D-12); stage 7 selects among them.
- **L-09:** model level, stage 9.
- **L-10 (GEFS) and L-11c:** outside the scope (volume).
- **L-11b (CAMS):** waits for the user (Q-14).

### Stage 7: feature selection (D-13)

| ID | Idea | Source | Pri. |
|---|---|---|---|
| L-12 | **Expect NWP features to dominate** and lags and calendar features to drop out (Q26, Q28). **Select per model family:** tabular models need the engineered temporal features, sequence models less so (C1). Check collinearity among NWP features (C2: collinear inputs hurt trees) | C1, C2, S1 | M |

### Stage 8: evaluation (§6.9)

| ID | Idea | Source | Pri. |
|---|---|---|---|
| L-13 | **CLIPER skill score** next to persistence skill: RMSE_cp = √(1 − γ²(h))·σ per target and horizon. γ is the lag-(24·h) autocorrelation of the daytime series, σ its standard deviation, **both from train**. Persistence skill stays for comparability with the original | S3, C3 | H |
| L-14 | **Potential RMSE skill** for models not trained on MSE (BiLSTM quantile loss): s = 1 − √(1 − ρ²)/√(1 − γ²(h)). Reported next to the actual skill; the verdict uses the metric each model optimises | C3 | M |
| L-15 | **Distribution-oriented diagnostics:** MBE, correlation, variance ratio V(f)/V(y), and conditional bias by forecast value, hour and season. An under-dispersed MSE-optimal forecast is expected (S2, C2) and is not a bug | S3, S2, C2 | M |
| L-16 | **Second normalisation:** nRMSE by mean daytime power, as in the literature (C1, C2), next to the capacity normalisation (3,449 kW, kept for comparability with the original). Stage 3b showed the peak is only 85 % of capacity (A-08) | C1, C2, A-08 | H |

**Status after stage 8B (2026-10-25):**
- **L-13 adopted** as an extra column, with σ and γ from the evaluated sample. With train
  statistics, plain persistence beat the reference on validation
  (`docs/stage8b_plan.md` §7).
- **L-16 adopted** (nRMSE by the mean actual AC of the evaluated rows).
- **L-14 and L-15** are produced in stage 11, with the error analysis (§12.3).

### Stages 9–10: models (D-08, D-17)

| ID | Idea | Source | Pri. |
|---|---|---|---|
| L-17 | **Light model zoo for development:** add Extra Trees (fast; best cost/accuracy with hybrid features) and an MLP or kernel ridge next to XGB, plus the physical baseline from L-01. **Same tuning protocol for all** (C1: boosting is sensitive to the budget) | C1, S1 | M |
| L-18 | **Blending:** equal-weight or validation-fitted weights for XGB + BiLSTM (+ physical). Worth about as much as the spread between models in C1. ML and physical errors differ (C2). Weights fixed on validation before the test is opened | C1, C2 | M |
| L-19 | **Neural models (input to D-08):** in C1, RNNs are the best single deep model and transformers are not better but much costlier. **A univariate PatchTST (the target only, §6.11) cannot use the weather forecast**, which is where day-ahead skill comes from, so it is expected to be weak. Check in the stage-9 smoke test whether neuralforecast's PatchTST really uses future exogenous inputs. If not, either report PatchTST as a univariate benchmark, or ask the user to revise D-08 | C1 | H (decision input) |

### Stage 11: reporting

| ID | Idea | Source | Pri. |
|---|---|---|---|
| L-20 | **Report errors on dust days and by season** (dust days identified by CAMS AOD or by the measured clear-hour deficit of L-11). Compare with the literature only after aligning the normalisation, and state the differences of climate, NWP and resolution | C4, S4, C1 | M |

### G-1 (audit)

- **Every new external source gets the leakage rule of T-17** (issue time + publication delay
  ≤ T0) and a perturbation test (T-09): CAMS, GEFS and GEFS-Aerosols.
- **The site-derived dust predictor (L-11a)** uses only day D−1, complete before T0.

## 5. What we expect, given the literature

- **Normalisation:** literature errors are normalised by mean power. In Hungary, nRMSE is
  ≈ 37–49 % of mean power (C1, C2). At Ketura, with mostly clear skies, errors relative to
  mean power should be much lower.
- **Skill:** skill against persistence will be lower than in Hungary in summer, because
  persistence is already good in clear weather. The CLIPER skill (L-13) is the fair measure.
- **Where the gains are:** in winter and spring (clouds, Q27) and on dust days (L-11).
- **Model choice:** the spread between XGB and the neural models should be small (C1). A
  combination (L-18) is more likely to help than a bigger network.
