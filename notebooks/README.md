# Notebooks

One notebook per stage. Logic lives in `src/solarpv/`; notebooks import it and explain each step in academic English.

| Notebook | Stage | Module |
|---|---|---|
| `01_data_ingest.ipynb` | 2 | `solarpv.data.ingest` |
| `02_eda.ipynb` | 3 | — |
| `03_preprocessing.ipynb` | 4 | `solarpv.data.preprocess` |
| `04_cleaning.ipynb` | 5 | `solarpv.data.clean` |
| `04b_features_target.ipynb` | 6 | `solarpv.features.build`, `solarpv.features.target` |
| `04c_nwp_gfs.ipynb` | 6b | `solarpv.nwp.gfs` |
| `05_feature_selection.ipynb` | 7 | — |
| `06_models.ipynb` | 9, 10 | `solarpv.models` |
| `07_evaluation.ipynb` | 8, 11 | `solarpv.evaluation` |
| `08_reporting.ipynb` | 11 | — |
