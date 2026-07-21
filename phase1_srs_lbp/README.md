# Phase 1 — SRS-LBP and PCA for Writer Identification

## Overview

This repository contains the Phase 1 implementation developed for the MSc thesis **Writer Identification**.

The method is evaluated on the ICDAR2017 Historical-WI dataset using binarised handwritten document images.

This repository is the locked presentation snapshot. See `SNAPSHOT_STATUS.md` for the reference status and `results/phase1_reference_metrics.json` for the recorded metrics.

## Final configuration

- TRAIN: 1,182 binarised pages from 394 writers
- TEST: 3,600 binarised pages from 720 writers
- SRS-LBP radii: 1–12
- Sampling points per radius: 8
- Initial descriptor dimensionality: 3,072
- L1 normalisation per histogram block
- Zero-pattern bin retained but set to zero
- PCA dimensionality reduction: 3,072 to 200
- PCA fitted only on TRAIN
- PCA whitening disabled
- Signed square-root normalisation
- Final L2 normalisation
- Within-TEST retrieval
- Squared Euclidean distance

## Repository structure

```text
.
├── README.md
├── requirements_phase1.txt
├── src/
│   ├── icdar2017_srs_lbp_binarized_10022026.py
│   ├── srs_lbp.py
│   ├── pca_and_normalize.py
│   └── evaluation.py
└── manifests/
    ├── manifest_train_1182.csv
    └── manifest_test_3600.csv
	
```

## Python files

- icdar2017_srs_lbp_binarized_10022026.py
Main execution script. It loads the manifests, extracts descriptors, fits PCA on TRAIN, transforms TRAIN and TEST, and performs retrieval evaluation on TEST. 
- srs_lbp.py
Implements the Sparse Radial Sampling Local Binary Pattern descriptor. 
- pca_and_normalize.py
Implements PCA projection, signed square-root normalisation and L2 normalisation. 
- evaluation.py
Implements squared Euclidean distances, Top-1 accuracy and mean Average Precision. 

## Dependencies

- Python 3.10 or newer.
- NumPy.
- pandas.
- Pillow.
- joblib.
- tqdm.
- scikit-learn.

## Installation

Python 3.10 or newer is recommended.
```
pip install -r requirements_phase1.txt 
```

## Dataset

The Historical-WI images are not included in this repository.

The ICDAR2017 Historical-WI dataset can be downloaded from Zenodo:
https://zenodo.org/records/1324999

The user must provide the local directories containing:
- the 1,182 binarised TRAIN pages; 
- the 3,600 binarised TEST pages. 

The following values must be edited inside the main script:
```
TRAIN_BASE_DIR = r"REPLACE_WITH_PATH_TO_TRAIN_BINARISED_IMAGES"
TEST_BASE_DIR = r"REPLACE_WITH_PATH_TO_TEST_BINARISED_IMAGES"
```

## Manifest format

The manifests have the following columns:
```
path,writer_id,split
```
Example:
```
1-IMG_MAX_10002.jpg,1,test
```
The path value is relative to the corresponding TRAIN or TEST image directory.

## Execution

Run the program from the repository root:
```
python src/icdar2017_srs_lbp_binarized_10022026.py
```
If previously generated descriptor files are found in the output directory, the script loads them instead of repeating SRS-LBP extraction.

## Generated outputs

The script generates:
- X_train_srs_lbp.npy 
- X_test_srs_lbp.npy 
- y_train_writer_id.npy 
- y_test_writer_id.npy 
- Z_train_pca_norm.npy 
- Z_test_pca_norm.npy 
- phase1_run_log.json 
- phase1_run_summary.txt 
These generated files are not included in the GitHub repository.

## Accepted Phase 1 results

- Top-1 accuracy: 74.14% 
- mAP: 42.08% 

## References

A. Nicolaou, A. D. Bagdanov, M. Liwicki, and D. Karatzas, “Sparse Radial Sampling LBP for Writer Identification,” ICDAR, 2015.
S. Fiel et al., “ICDAR2017 Competition on Historical Document Writer Identification (Historical-WI),” ICDAR, 2017.

