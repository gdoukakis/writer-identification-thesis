# Writer Identification - MSc Thesis Code

This repository contains the code developed for the MSc thesis **Writer Identification**.

The thesis includes two main experimental phases:

## Phase 1 - SRS-LBP Baseline

This phase implements a handcrafted writer-identification baseline based on SRS-LBP features and PCA-based retrieval.

Folder:

    phase1_srs_lbp/

The Phase 1 code will be added in this folder.

## Phase 2 - Christlein-faithful Pipeline

This phase contains the final accepted deep local-feature pipeline, implemented as a faithful re-implementation of the Christlein et al. (2017) writer identification pipeline.

Folder:

    phase2_christlein2017/

Accepted thesis results:

| Method | Top-1 | mAP |
|---|---:|---:|
| Direct cosine retrieval | 86.69% | 70.28% |
| Exact E-SVM-FE | 87.47% | 71.70% |

## Dataset

The ICDAR2017 Historical-WI dataset is not included in this repository.

Expected protocol:

| Split | Images | Writers |
|---|---:|---:|
| TRAIN | 1,182 | 394 |
| TEST | 3,600 | 720 |

## Repository structure

    Repository structure

    writer-identification-thesis/
    ├── README.md
    ├── .gitignore
    ├── phase1_srs_lbp/
    │   └── README_PHASE1.md
    └── phase2_christlein2017/
        ├── README_PHASE2.md
        ├── requirements_phase2.txt
        ├── configs/
        ├── src/
        └── scripts/

## Notes

This repository does not include datasets, extracted descriptors, extracted patches, trained checkpoints, intermediate outputs, logs, or result artefacts.

The Phase 2 implementation should be described as a strict Christlein-faithful re-implementation of the published methodology, but not as a bitwise or source-code-level reproduction of the original author implementation.

