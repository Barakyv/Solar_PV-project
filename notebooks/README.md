# Notebooks

One notebook per stage, in the numbering of the original project. Logic lives in `src/solarpv/`;
notebooks import it and explain each step in academic English. Every code cell is followed by a
markdown note (what it shows, what it means, what was decided); the numbers in the notes are read
from the stage outputs by the builder script, never typed. Each notebook is written and executed
by its builder in `scripts/`, so it can be shown live and rebuilt after any rerun.

| Notebook | Stage | Builder | Module |
|---|---|---|---|
| `00b_original_backup_review.ipynb` | audit (2026-11-01) | `make_original_backup_review_notebook.py` | `00_data_audit/original_backup_2026-11-01/` |
| `01_data_ingest.ipynb` | 2 | `make_data_ingest_notebook.py` | `solarpv.data.ingest` |
| `02_eda.ipynb` | 3 | `make_eda_notebook.py` | `solarpv.eda` |
| `02b_data_quality.ipynb` | 3b | `make_quality_notebook.py` | `solarpv.eda.quality` |
| `03_preprocessing.ipynb` | 4 | `make_preprocessing_notebook.py` | `solarpv.data.preprocess` |
| `04_cleaning.ipynb` | 5 | `make_cleaning_notebook.py` | `solarpv.data.clean` |
| `04b_features_target.ipynb` | 6 | `make_features_notebook.py` | `solarpv.features.build`, `solarpv.features.target` |
| `04c_nwp_gfs.ipynb` | 6b | `make_nwp_gfs_notebook.py` | `solarpv.nwp` |
| `05_feature_selection.ipynb` | 7 | to come | — |
| `06_models.ipynb` | 9, 10 | to come | `solarpv.models` |
| `07_evaluation.ipynb` | 8 (11 later) | `make_evaluation_notebook.py` | `solarpv.evaluation` |
| `08_reporting.ipynb` | 11 | to come | — |

The builders added on 2026-11-01 share `scripts/_notebook.py`, which records the sha256 of every
source file in the notebook metadata; `tests/contracts/test_stage_notebooks.py` fails when a
source changed and the notebook was not rebuilt.
