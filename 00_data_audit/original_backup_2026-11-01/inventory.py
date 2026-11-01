"""Inventory of the original project backup, including what survives only in its git history.

    python 00_data_audit/original_backup_2026-11-01/inventory.py

Writes `inventory.csv` (one row per file of the working copy and per git blob that matches no
working-copy file) and `inventory_summary.json`. Blobs other than images are written to
`data/external/original_backup_recovered/` (git-ignored) under their blob id, so that the other
audit scripts and the review notebook can read them. Read-only on the backup.
"""

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import _backup as B
import pandas as pd

from solarpv.config.runlog import RunLog

HERE = Path(__file__).resolve().parent
SEL = "feature selection: base stage that fed the submitted 05e"
SEL_OLD = "feature selection: earlier version (Dec 2025)"
WX = "weather source of the original NWP"
MOD = "modelling: intermediate"
FE = "feature engineering: intermediate"
EDA = "EDA, ingest, preprocessing: superseded by stages 2-5"
RAW = "raw export copies (same raw data as `Raw data/`)"

# (substring of the label, group); first match wins. The label of a working-copy file is its path
# and notebook title; the label of a history blob is its title and its first characters (or, for
# a notebook, its code).
GROUPS = [
    ("05b_feature_selection_direct", SEL), ("Feature Selection - Indirect", SEL),
    ("artifacts/05_feature_selection", SEL), ("outputs/05_feature_selection", SEL),
    ("Feature Selection and Optimization", SEL_OLD),
    ("Notebook: 05_feature_selection.ipynb", SEL_OLD),
    ("http_cache", WX), ("ketura_gfs_forecast", WX),
    ('"learner"', MOD), ('"n_estimators"', MOD), ('"loss"', MOD), ('"lookback_hours"', MOD),
    ("EXPERIMENT", MOD), ("Season,Model", MOD), ("StandardScaler", MOD),
    ("06_modeling", MOD), ("models/", MOD), ("07_lstm", MOD), ("XGBoost Modeling", MOD),
    ("LSTM", MOD), ("Direct vs Indirect", MOD), ("Check if model was saved", MOD),
    ("06 - Modeling", MOD),
    ("multihorizon", FE), ("Multi-Horizon", FE), ("04_feature_engineering", FE),
    ("Feature Engineering", FE), ("Feature Builder", FE),
    ("02_eda", EDA), ("outputs/EDA", EDA), ("EDA", EDA), ("01_data_ingest", EDA),
    ("Data Ingest", EDA), ("Data Ingestion", EDA), ("Readiness Validator", EDA),
    ("03_preprocessing", EDA), ("decisions_cleaning", EDA),
    ("XGBRegressor", MOD),
    ("Sites/PVS/Ktora", RAW), ("t_stamp,", EDA), ("timestamp,", EDA),
]
DEFAULTS = {"png": "figures", "parquet": "data tables: intermediate", "sqlite": WX,
            "zip (keras / torch)": MOD, "text": EDA}
# code of the final v6 pipeline (PROJECT_MASTER.md §6.10, R-16): is any of it in the backup?
FINAL_CODE = {
    "xgb_sota.py": r"xgb_sota",
    "lstm_sota.py (PyTorch encoder-decoder)": r"lstm_sota\.py|EncoderDecoder",
    "TFT (pytorch-forecasting)": r"TemporalFusionTransformer|pytorch_forecasting",
    "PatchTST": r"PatchTST",
    "05c / 05e / 05f selection code": r"elbow_|stability_selection|consensus_2of3|_05f",
}
USED_IN_AUDIT = {"05b_feature_selection_direct.ipynb", "17d0183ad2b9", "http_cache.sqlite",
                 "bb194fca9d7e", "ketura_gfs_forecast.parquet", "stage4_shap_importance_direct.csv",
                 "selected_features_direct.json", "xgb_physics_q50.json", "0756fbeea89a"}


# for a history notebook whose title names no stage: what its code does
CODE_RULES = [("XGBRegressor", MOD), ("nn.LSTM", MOD), ("keras", MOD), ("pvlib", FE),
              ("read_csv", EDA)]


def group_of(label: str, kind: str, code: str = "") -> str:
    """The group of the first rule found in `label`, else in the notebook `code`, else by kind."""
    for rules, text in ((GROUPS, label), (CODE_RULES, code)):
        for key, group in rules:
            if key in text:
                return group
    return DEFAULTS.get(kind, "configuration, IDE and other")


def code_text(body: bytes) -> str:
    nb = json.loads(body.decode("utf-8"))
    return "\n".join("".join(c.get("source", "")) for c in nb.get("cells", [])
                     if c.get("cell_type") == "code")


def main() -> int:
    with RunLog("audit", "python 00_data_audit/original_backup_2026-11-01/inventory.py",
                out_dir=HERE) as log:
        B.RECOVERED.mkdir(parents=True, exist_ok=True)
        rows, code_corpus = [], []
        with log.step("working copy"):
            work = B.worktree_files()
            for rel in work.values():
                body = (B.BACKUP / rel).read_bytes()
                kind = B.kind_of(body)
                title = B.notebook_title(body) if kind == "notebook" else ""
                if kind == "notebook":
                    code_corpus.append(code_text(body))
                elif rel.endswith(".py"):
                    code_corpus.append(body.decode("utf-8", "replace"))
                rows.append({"location": "working copy", "item": rel, "kind": kind,
                             "size_bytes": len(body), "title": title,
                             "group": group_of(rel + " " + title, kind),
                             "used_in_audit": any(u in rel for u in USED_IN_AUDIT)})
            log.note(f"{len(work)} files in the working copy")
        with log.step("git history"):
            kinds, extra = Counter(), 0
            for sha, kind, body in B.loose_objects():
                kinds[kind] += 1
                if kind != "blob" or sha in work:
                    continue
                extra += 1
                k = B.kind_of(body)
                title, label, code = "", "", ""
                if k == "notebook":
                    title = B.notebook_title(body)
                    code = code_text(body)
                    code_corpus.append(code)
                    label = title
                elif k in ("text", "json"):
                    label = body[:300].decode("utf-8", "replace")
                elif k == "binary":
                    label = body[:300].decode("latin-1")
                ext = {"notebook": ".ipynb", "json": ".json", "parquet": ".parquet",
                       "sqlite": ".sqlite", "text": ".txt"}.get(k, ".bin")
                if k != "png":
                    (B.RECOVERED / f"{sha[:12]}{ext}").write_bytes(body)
                rows.append({"location": "git history only", "item": sha[:12], "kind": k,
                             "size_bytes": len(body), "title": title,
                             "group": group_of(label, k, code),
                             "used_in_audit": sha[:12] in USED_IN_AUDIT})
            log.note(f"loose objects {dict(kinds)}; {extra} blobs match no working-copy file")
        inv = pd.DataFrame(rows).sort_values(["group", "location", "item"])
        inv.to_csv(HERE / "inventory.csv", index=False, lineterminator="\n")

        commits = []
        for line in (B.BACKUP / ".git/logs/refs/heads/master").read_text().splitlines():
            meta, _, message = line.partition("\t")
            stamp = int(meta.split()[-2])
            commits.append({"date_utc": datetime.fromtimestamp(stamp, UTC).date().isoformat(),
                            "message": message})
        corpus = "\n".join(code_corpus)
        summary = {
            "backup": "old files/solar_ds_capstone",
            "status": "an older backup (about January 2026), a step on the way to the v6 "
                      "submission; never treated as the final pipeline",
            "git": {"objects_by_kind": dict(kinds), "commits_in_reflog": commits,
                    "missing": "HEAD, config, index, the root tree"},
            "files": {loc: int(n) for loc, n in inv["location"].value_counts().items()},
            "bytes": {loc: int(n)
                      for loc, n in inv.groupby("location")["size_bytes"].sum().items()},
            "by_group": {g: int(n) for g, n in inv["group"].value_counts().items()},
            "notebooks": inv.loc[inv["kind"] == "notebook", ["location", "item", "title", "group"]]
            .to_dict("records"),
            "notebooks_searched": len(code_corpus),
            "final_code_present": {name: bool(re.search(pattern, corpus))
                                   for name, pattern in FINAL_CODE.items()},
        }
        (HERE / "inventory_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
            newline="\n")
        for group, n in summary["by_group"].items():
            log.note(f"{n:4d}  {group}")
        log.note(f"final v6 code found: {summary['final_code_present']}")
        log.outputs([HERE / "inventory.csv", HERE / "inventory_summary.json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
