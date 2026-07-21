# Phase 2 Snapshot Status

This folder contains the final accepted Phase 2 code snapshot used in the MSc thesis **Writer Identification**.

Final branch:

    strict_christlein2017_rsift_random500k

The implementation corresponds to the final accepted Christlein et al. pipeline re-implementation used for the thesis results.

## Included

- Source code for Stages 01R-07R
- Helper modules
- Configuration example
- Dataset manifests
- Dependency file
- README documentation

## Excluded

The following files are intentionally not included in the GitHub repository:

- Historical-WI images
- extracted patches
- raw descriptors
- local learned embeddings
- trained checkpoints
- m-VLAD codebooks
- PCA models
- E-SVM-FE generated outputs
- runtime logs
- intermediate output artefacts

## Accepted thesis results

| Method | Top-1 | mAP |
|---|---:|---:|
| Direct cosine retrieval | 86.69% | 70.28% |
| Exact E-SVM-FE | 87.47% | 71.70% |
