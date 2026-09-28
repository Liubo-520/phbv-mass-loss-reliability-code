# PHBV mass-loss reliability analysis: research code

This repository contains the Python analysis code for the study **Environmental Transferability of PHBV Mass-Loss Trajectories: Study-Aware Evaluation and Physically Constrained Prediction**. It reconstructs decimal-precision PHBV mass-loss observations, saves row-, curve-, and study-separated splits, evaluates traditional machine-learning models, and computes uncertainty and figure/table outputs. The endpoint is measured gravimetric mass loss, not ultimate biodegradation or mineralization.

## Source data

The original workbook and cleaned CSV belong to Kotzabasaki et al. (2026), [*Polymers* 18, 897](https://doi.org/10.3390/polym18070897). Download the two files from the [authors' public repository](https://github.com/FSL-AUA/Weight-loss-model) and put them at:

```text
origin_data/Weight-loss-model-main/scl_weight_loss_data_library_raw_LM.xlsx
origin_data/Weight-loss-model-main/weight_loss_data_LM.csv
```

The source files are deliberately absent from this repository. `origin_data/SHA256SUMS.txt` records the expected hashes for these two inputs; `scripts/01_audit_data.py` verifies them before analysis. If the upstream files change, retrieve the version matching those hashes. The source article and notebooks are not required by the executable pipeline.

## Reproduce the analyses

Use Python 3.10 from the repository root. Install the pinned environment with `python -m pip install -r requirements-lock.txt`. Run these scripts in order; they write reconstructed data to `data/`, fixed splits to `splits/`, and predictions, metrics, and figures to `results/`, `figures/`, and `tables/`.

```text
python scripts/01_audit_data.py
python scripts/02_reconstruct_dataset.py
python scripts/03_generate_splits.py
python scripts/04_run_screening.py
python scripts/05_run_nested_loso.py --trials 12
python scripts/06_run_framework_ablations.py
python scripts/07_run_uncertainty.py
python scripts/08_run_statistics.py
python scripts/09_train_final_model.py
python scripts/10_run_interpretation.py
python scripts/20_run_revision_analyses.py
python scripts/21_run_endpoint_audit.py
python scripts/22_run_simple_loso_baselines.py
python scripts/23_run_corpus_endpoint_sensitivities.py
python scripts/11_generate_figures_tables.py
python -m pytest -q
```

The nested study-level runs can take substantial CPU time. Model selection and preprocessing are fitted within the appropriate training studies; random-row results are only an interpolation comparison. Some publication layout assets and manuscript-building tools are outside this code release. The research-use predictor is available separately as a [Hugging Face Space](https://huggingface.co/spaces/AI4deeperScience/phbv-mass-loss-reliability).

## License and provenance

Project-authored code is released under the MIT License. The upstream data and article retain their original rights. The two source-file hashes identify the exact inputs used for this study; no source data or credentials are included here.
