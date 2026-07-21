# Writer Identification - MSc Thesis Code

This repository contains the code developed for the MSc thesis **Writer Identification**.

The thesis includes two main experimental phases:

## Phase 1 - SRS-LBP Baseline

This phase contains the handcrafted baseline used in the thesis. It implements SRS-LBP feature extraction, PCA-based dimensionality reduction and retrieval-based writer identification on the ICDAR2017 Historical-WI dataset.

Folder:

    phase1_srs_lbp/

Accepted thesis results:

| Method | Top-1 | mAP |
|---|---:|---:|
| SRS-LBP + PCA | 74.14% | 52.57% |

The Phase 1 folder includes the source code, manifests, reference metrics, snapshot information and dependencies required to reproduce the baseline.

## Phase 2 - Christlein et al. Pipeline Re-implementation

This phase contains the final accepted deep local-feature pipeline. It is implemented as a re-implementation of the Christlein et al. (2017) writer identification pipeline.

Folder:

    phase2_christlein2017/

Accepted thesis results:

| Method | Top-1 | mAP |
|---|---:|---:|
| Direct cosine retrieval | 86.69% | 70.28% |
| Exact E-SVM-FE | 87.47% | 71.70% |

The Phase 2 folder includes the source code, scripts, configuration example, dependencies and running instructions for the final accepted pipeline.

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

## Repository structure

    writer-identification-thesis/
    ├── README.md
    ├── .gitignore
    ├── phase1_srs_lbp/
    │   ├── README.md
    │   ├── SNAPSHOT_STATUS.md
    │   ├── SHA256SUMS.txt
    │   ├── requirements_phase1.txt
    │   ├── .gitignore
    │   ├── manifests/
    │   │   ├── manifest_train_1182.csv
    │   │   └── manifest_test_3600.csv
    │   ├── results/
    │   │   └── phase1_run_summary.txt
    │   └── src/
    │       ├── evaluation.py
    │       ├── icdar2017_srs_lbp_binarized_10022026.py
    │       ├── pca_and_normalize.py
    │       └── srs_lbp.py
    └── phase2_christlein2017/
        ├── README.md
        ├── SNAPSHOT_STATUS.md
        ├── SHA256SUMS.txt
        ├── requirements_phase2.txt
        ├── .gitignore
        ├── configs/
        │   └── phase2_strict_rsift_random500k.example.yaml
        ├── manifests/
        │   ├── manifest_train_1182.csv
        │   └── manifest_test_3600.csv
        ├── scripts/
        │   └── christlein2017_faithful/
        └── src/
            └── christlein2017_faithful/

## Notes

This repository does not include datasets, extracted descriptors, extracted patches, trained checkpoints, intermediate outputs, logs, or result artefacts.

The Phase 2 implementation should be described as a strict re-implementation of the published methodology of Christlein et al., but not as a bitwise or source-code-level reproduction of the original author implementation.
