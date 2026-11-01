"""Evidence on the original feature selection, from the backup (feeds F-26, F-27 and stage 7).

    python 00_data_audit/original_backup_2026-11-01/inventory.py        # recovers the blobs
    python 00_data_audit/original_backup_2026-11-01/selection_evidence.py

Checks, all read-only:
1. **Which stage fed the submitted 05e files.** The two base-selection notebooks of January 2026
   (direct: working copy; indirect: git history) end with 42 features. Are these the 42 starting
   features of the submitted D+1 consensus files? For the indirect route the selection JSON is
   not in the backup; the 42 features of the intermediate XGBoost trained on it stand in.
2. **Their configuration** (parsed with `ast.literal_eval`, never executed): forbidden list,
   protected core list, category minimums, thresholds. Compared with the earlier notebook 05
   (December 2025), whose parameters PROJECT_MASTER.md §7 had taken.
3. **Leakage in the direct selection:** the per-inverter DC power of the same hour is not in the
   forbidden list; its share of the SHAP importance.
4. **Rows used:** any filter by split or date before the selection; how gaps are filled.
"""

import ast
import json
import re
from pathlib import Path

import _backup as B
import pandas as pd

from solarpv.config.runlog import RunLog

HERE = Path(__file__).resolve().parent
DIRECT_NB = B.BACKUP / "05_feature_selection/old/05b_feature_selection_direct.ipynb"
INDIRECT_NB = "17d0183ad2b9"
EARLIER_NB = "0756fbeea89a"  # identical to old files/05_feature_selection.ipynb
DIRECT_OUT = B.BACKUP / "artifacts/05_feature_selection/selected_features_direct.json"
INDIRECT_MODEL = B.BACKUP / "outputs/06_modeling/models/final_checkpoints/xgb_physics_q50.json"
SHAP_DIRECT = B.BACKUP / "outputs/05_feature_selection/reports/stage4_shap_importance_direct.csv"
E05 = B.SUBMISSION / "09_feature_selection"
ROW_FILTER = r"\[['\"]split['\"]\]|train_end|TRAIN_END|2024-0[12]|\.loc\[\s*['\"]20|query\("


def config_of(code_cells: list[str]) -> dict:
    for code in code_cells:
        start = code.find("CONFIG = {")
        if start >= 0:
            end = code.find("\n}", start)
            return ast.literal_eval(code[start + len("CONFIG = "):end + 2])
    raise ValueError("no CONFIG literal")


def earlier_params(code_cells: list[str]) -> dict:
    text = "\n".join(code_cells)

    def num(key):
        return float(re.search(rf"'{key}':\s*([0-9.]+)", text).group(1))

    block = re.search(r"'core_features':\s*\{(.*?)\n    \},", text, flags=re.S).group(1)
    core = [name for group in re.findall(r"\[(.*?)\]", block, flags=re.S)
            for name in re.findall(r"'([a-z0-9_]+)'", group)]
    return {"variance_threshold": num("variance_threshold"), "mi_threshold": num("mi_threshold"),
            "core_features": core, "rfe_xgb_max_depth": int(num("max_depth"))}


def rows_and_gaps(code_cells: list[str]) -> dict:
    text = "\n".join(code_cells)
    return {"reads_whole_table": "pd.read_parquet(data_file)" in text,
            "row_filter_by_split_or_date": bool(re.search(ROW_FILTER, text)),
            "time_series_cv_over_all_rows": "TimeSeriesSplit" in text,
            "gap_fill_whole_column_median": "fillna(df[col].median())" in text,
            "gap_fill_zero_before_shap": ".fillna(0)" in text}


def start_sets() -> dict:
    out = {}
    for target in ("direct", "indirect"):
        d1 = None
        for h in (1, 2, 3):
            e = json.loads((E05 / f"final_features_{target}_D{h}.json").read_text(encoding="utf-8"))
            start = e["methods"]["shap_elbow"]["shap_rank"]
            d1 = d1 or start
            out[f"{target}_D{h}"] = {"n_start": len(start), "subset_of_D1": set(start) <= set(d1),
                                     "dropped_from_D1": sorted(set(d1) - set(start))}
    return out


def main() -> int:
    with RunLog("audit", "python 00_data_audit/original_backup_2026-11-01/selection_evidence.py",
                out_dir=HERE) as log:
        indirect_path = B.recovered(INDIRECT_NB)
        earlier_path = B.recovered(EARLIER_NB)
        log.inputs([DIRECT_NB, indirect_path, earlier_path, DIRECT_OUT, INDIRECT_MODEL,
                    SHAP_DIRECT] + sorted(E05.glob("final_features_*_D1.json")))
        notebooks = {"direct": B.notebook_code(DIRECT_NB),
                     "indirect": B.notebook_code(indirect_path)}
        e05_d1 = {t: json.loads((E05 / f"final_features_{t}_D1.json").read_text(encoding="utf-8"))
                  ["methods"]["shap_elbow"]["shap_rank"] for t in ("direct", "indirect")}
        outputs = {
            "direct": json.loads(DIRECT_OUT.read_text(encoding="utf-8"))["selected_features"],
            "indirect": json.loads(INDIRECT_MODEL.read_text(encoding="utf-8"))
            ["learner"]["feature_names"]}
        evidence = {"base_notebooks": {}, "fed_submitted_05e_D1": {}}
        for target, cells in notebooks.items():
            cfg = config_of(cells)
            forbidden = cfg["forbidden_features"]
            evidence["base_notebooks"][target] = {
                "target": cfg["target"], "forbidden_features": forbidden,
                "per_inverter_dc_forbidden": all(c in forbidden for c in
                                                 ("s01_inv1_DC_kW", "s01_inv2_DC_kW")),
                "core_features": cfg["core_features"], "category_min": cfg["category_min"],
                "variance_threshold": cfg["variance_threshold"],
                "mi_threshold": cfg["mi_threshold"], "target_n_features": cfg["target_n_features"],
                "cv_splits": cfg["cv_splits"], **rows_and_gaps(cells)}
            same = set(outputs[target]) == set(e05_d1[target])
            evidence["fed_submitted_05e_D1"][target] = {
                "n_base_output": len(outputs[target]), "n_05e_start": len(e05_d1[target]),
                "identical_sets": same,
                "base_output_source": "selected_features_direct.json" if target == "direct"
                else "feature names of the intermediate XGBoost xgb_physics_q50.json"}
            log.note(f"{target}: base output {len(outputs[target])} features, identical to the "
                     f"submitted 05e D+1 start: {same}; per-inverter DC forbidden: "
                     f"{evidence['base_notebooks'][target]['per_inverter_dc_forbidden']}")
        evidence["submitted_05e_start_sets"] = start_sets()
        evidence["earlier_notebook_05_dec2025"] = earlier_params(B.notebook_code(earlier_path))

        shap = pd.read_csv(SHAP_DIRECT).sort_values("mean_abs_shap", ascending=False)
        top = shap.head(5).reset_index(drop=True)
        top.to_csv(HERE / "selection_shap_direct_top.csv", index=False, lineterminator="\n")
        evidence["direct_shap"] = {
            "top5": top[["feature", "mean_abs_shap"]].round(6).to_dict("records"),
            "dc_share_of_total": round(float(shap.loc[shap["feature"].str.contains("_DC_kW"),
                                                      "mean_abs_shap"].sum()
                                             / shap["mean_abs_shap"].sum()), 4),
            "first_over_third": round(float(top.loc[0, "mean_abs_shap"]
                                            / top.loc[2, "mean_abs_shap"]), 1)}
        log.note(f"direct SHAP: DC share {evidence['direct_shap']['dc_share_of_total']}, "
                 f"first / third {evidence['direct_shap']['first_over_third']}")
        out = HERE / "selection_evidence.json"
        out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
        log.outputs([out, HERE / "selection_shap_direct_top.csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
