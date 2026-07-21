# Phase 1 — SRS-LBP Baseline for Writer Identification

## Overview

This folder contains the Phase 1 implementation developed for the MSc thesis **Writer Identification**.

Phase 1 implements a handcrafted writer-identification baseline based on Sparse Radial Sampling Local Binary Patterns (SRS-LBP), PCA-based dimensionality reduction and retrieval-based evaluation on the ICDAR2017 Historical-WI dataset.

This folder corresponds to the locked Phase 1 presentation snapshot. See `SNAPSHOT_STATUS.md` for the snapshot status and `results/phase1_reference_metrics.json` for the recorded reference metrics.

## Final configuration

- Dataset: ICDAR2017 Historical-WI
- Image type: binarised handwritten document pages
- TRAIN split: 1,182 pages from 394 writers
- TEST split: 3,600 pages from 720 writers
- Descriptor: SRS-LBP
- SRS-LBP radii: 1–12
- Sampling points per radius: 8
- Initial descriptor dimensionality: 3,072
- Histogram normalisation: L1 normalisation per histogram block
- Zero-pattern handling: zero-pattern bin retained but set to zero
- Dimensionality reduction: PCA from 3,072 to 200 dimensions
- PCA fitting source: TRAIN only
- PCA whitening: disabled
- Post-processing: signed square-root normalisation followed by L2 normalisation
- Retrieval protocol: within-TEST retrieval
- Distance metric: squared Euclidean distance
- Self-match handling: self-match excluded

## Accepted thesis results

| Method | Top-1 | mAP |
|---|---:|---:|
| SRS-LBP + PCA | 74.14% | 42.08% |

## Repository structure

    phase1_srs_lbp/
    ├── README.md
    ├── SNAPSHOT_STATUS.md
    ├── SHA256SUMS.txt
    ├── requirements_phase1.txt
    ├── manifests/
    │   ├── manifest_train_1182.csv
    │   └── manifest_test_3600.csv
    ├── results/
    │   └── phase1_reference_metrics.json
    └── src/
        ├── evaluation.py
        ├── icdar2017_srs_lbp_binarized_10022026.py
        ├── pca_and_normalize.py
        └── srs_lbp.py

## Python files

- `src/icdar2017_srs_lbp_binarized_10022026.py`  
  Main execution script. It loads the manifests, extracts SRS-LBP descriptors, fits PCA on TRAIN, transforms TRAIN and TEST, and performs retrieval evaluation on TEST.

- `src/srs_lbp.py`  
  Implements the Sparse Radial Sampling Local Binary Pattern descriptor.

- `src/pca_and_normalize.py`  
  Implements PCA projection, signed square-root normalisation and L2 normalisation.

- `src/evaluation.py`  
  Implements squared Euclidean distances, Top-1 accuracy and mean Average Precision.

## Dependencies

Python 3.10 or newer is recommended.

Main dependencies:

- NumPy
- pandas
- Pillow
- joblib
- tqdm
- scikit-learn

## Installation

From the `phase1_srs_lbp/` folder:

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements_phase1.txt

## Dataset

The Historical-WI images are not included in this repository.

The ICDAR2017 Historical-WI dataset can be downloaded from Zenodo:

    https://zenodo.org/records/1324999

The user must provide the local directories containing:

- the 1,182 binarised TRAIN pages
- the 3,600 binarised TEST pages

Expected protocol:

| Split | Images | Writers |
|---|---:|---:|
| TRAIN | 1,182 | 394 |
| TEST | 3,600 | 720 |

## Input configuration

The following values must be edited inside the main script before execution:

    TRAIN_BASE_DIR = r"REPLACE_WITH_PATH_TO_TRAIN_BINARISED_IMAGES"
    TEST_BASE_DIR = r"REPLACE_WITH_PATH_TO_TEST_BINARISED_IMAGES"

The provided manifests use paths relative to the corresponding TRAIN or TEST image directory.

## Manifest format

The manifests have the following columns:

    path,writer_id,split

Example:

    1-IMG_MAX_10002.jpg,1,test

## Execution

Run the program from the `phase1_srs_lbp/` folder:

    python src/icdar2017_srs_lbp_binarized_10022026.py

If previously generated descriptor files are found in the output directory, the script loads them instead of repeating SRS-LBP extraction.

## Generated outputs

The script generates intermediate descriptors, transformed features and result summaries, including:

- `X_train_srs_lbp.npy`
- `X_test_srs_lbp.npy`
- `y_train_writer_id.npy`
- `y_test_writer_id.npy`
- `Z_train_pca_norm.npy`
- `Z_test_pca_norm.npy`
- `phase1_run_log.json`
- `phase1_run_summary.txt`

These generated files are not included in the GitHub repository.

## Notes

This folder contains code and lightweight metadata only. It does not include Historical-WI images, extracted descriptors, transformed features or generated runtime outputs.

The Phase 1 method is used in the thesis as a handcrafted baseline against which the final Phase 2 pipeline is compared.

## References

A. Nicolaou, A. D. Bagdanov, M. Liwicki, and D. Karatzas, “Sparse Radial Sampling LBP for Writer Identification,” ICDAR, 2015.

S. Fiel et al., “ICDAR2017 Competition on Historical Document Writer Identification (Historical-WI),” ICDAR, 2017.
