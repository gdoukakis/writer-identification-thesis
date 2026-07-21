# Phase 2 — Christlein et al. Pipeline Re-implementation for Writer Identification

## Overview

This folder contains the Phase 2 implementation developed for the MSc thesis **Writer Identification**.

Phase 2 implements the final accepted deep local-feature pipeline, following the main writer-identification methodology of Christlein et al. (2017) at method level. The implementation does not claim bitwise or source-code-level identity with the original author implementation.

The final branch used in the thesis is:

    strict_christlein2017_rsift_random500k

## Final configuration

- Dataset: ICDAR2017 Historical-WI
- Image type: binarised handwritten document pages
- TRAIN split: 1,182 pages from 394 writers
- TEST split: 3,600 pages from 720 writers
- Local feature extraction: R-SIFT-like / RootSIFT
- Patch size: 32 x 32
- Local descriptor reduction: RootSIFT PCA whitening from 128 to 32 dimensions
- Local descriptor sample: random sample of 500,000 TRAIN descriptors
- Surrogate visual classes: KMeans with K = 5,000 clusters
- Ratio threshold: 0.9
- Surrogate CNN: ResNet-20
- Learned local embedding dimensionality: 64
- Aggregation: m-VLAD
- Number of m-VLAD codebooks: 5
- Centres per codebook: 64
- TRAIN embeddings per codebook: 500,000
- Post-VLAD normalisation: global signed square-root normalisation followed by global L2 normalisation
- Final dimensionality reduction: PCA whitening from 20,480 to 640 dimensions
- Retrieval baseline: direct cosine retrieval
- Final encoding: exact Exemplar-SVM Feature Encoding
- Selected E-SVM C value: 3.0
- Retrieval protocol: TEST-only leave-one-image-out retrieval
- Self-match handling: self-match excluded

## Accepted thesis results

| Method | Top-1 | mAP |
|---|---:|---:|
| Direct cosine retrieval | 86.69% | 70.28% |
| Exact E-SVM-FE | 87.47% | 71.70% |

## Repository structure

    phase2_christlein2017/
    ├── README_PHASE2.md
    ├── requirements_phase2.txt
    ├── configs/
    │   └── phase2_strict_rsift_random500k.example.yaml
    ├── scripts/
    │   └── christlein2017_faithful/
    │       ├── 04r_resnet20_full_surrogate_training_rsift_random500k.py
    │       ├── 05r_extract_resnet20_embeddings_rsift_random500k.py
    │       ├── 06r_mvlad_fit_encode_resnet20_rsift_random500k.py
    │       ├── 06r_mvlad_encode_resnet20_rsift_random500k_global_ssr_l2.py
    │       ├── 06r_mvlad_pca_whiten_resnet20_rsift_random500k.py
    │       ├── 06r_retrieval_eval_resnet20_rsift_random500k_pca640.py
    │       └── 07r_exact_esvm_fe_resnet20_rsift_random500k_pca640.py
    └── src/
        └── christlein2017_faithful/
            ├── 01r_extract_rsift_rootsift_patches.py
            ├── 02r_fit_rsift_pca_kmeans_random500k.py
            ├── 03r_build_surrogate_patch_dataset_rsift_random500k.py
            ├── config.py
            ├── datasets.py
            ├── esvm_fe.py
            ├── io_utils.py
            ├── metrics.py
            ├── resnet20.py
            ├── resnet20_pre_activation.py
            ├── rpca.py
            └── vlad.py

## Python files

- `src/christlein2017_faithful/01r_extract_rsift_rootsift_patches.py`  
  Extracts R-SIFT-like / RootSIFT local descriptors and 32 x 32 patches.

- `src/christlein2017_faithful/02r_fit_rsift_pca_kmeans_random500k.py`  
  Fits RootSIFT PCA32 with whitening and KMeans surrogate visual classes using TRAIN-only sampled descriptors.

- `src/christlein2017_faithful/03r_build_surrogate_patch_dataset_rsift_random500k.py`  
  Builds the surrogate patch dataset using KMeans assignments and ratio-test filtering.

- `scripts/christlein2017_faithful/04r_resnet20_full_surrogate_training_rsift_random500k.py`  
  Trains the ResNet-20 surrogate CNN.

- `scripts/christlein2017_faithful/05r_extract_resnet20_embeddings_rsift_random500k.py`  
  Extracts 64-dimensional learned local embeddings from the trained ResNet-20 model.

- `scripts/christlein2017_faithful/06r_mvlad_fit_encode_resnet20_rsift_random500k.py`  
  Fits the m-VLAD codebooks and encodes TRAIN pages.

- `scripts/christlein2017_faithful/06r_mvlad_encode_resnet20_rsift_random500k_global_ssr_l2.py`  
  Encodes TRAIN and TEST pages using global signed square-root and global L2 normalisation.

- `scripts/christlein2017_faithful/06r_mvlad_pca_whiten_resnet20_rsift_random500k.py`  
  Fits TRAIN-only PCA whitening from 20,480 to 640 dimensions and transforms TRAIN and TEST descriptors.

- `scripts/christlein2017_faithful/06r_retrieval_eval_resnet20_rsift_random500k_pca640.py`  
  Runs direct cosine retrieval on the final PCA640 descriptors.

- `scripts/christlein2017_faithful/07r_exact_esvm_fe_resnet20_rsift_random500k_pca640.py`  
  Runs exact Exemplar-SVM Feature Encoding and final TEST-only retrieval evaluation.

- `src/christlein2017_faithful/*.py` helper modules  
  Provide dataset loading, metrics, ResNet-20 definitions, VLAD utilities, PCA utilities, I/O helpers and E-SVM-FE utilities.

## Dependencies

Python 3.10 or newer is recommended.

Main dependencies:

- NumPy
- SciPy
- scikit-learn
- pandas
- joblib
- tqdm
- Pillow
- matplotlib
- psutil
- threadpoolctl
- packaging
- OpenCV
- PyTorch

Install PyTorch separately according to the target CUDA setup.

## Installation

From the `phase2_christlein2017/` folder:

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements_phase2.txt

For GPU execution, install the appropriate PyTorch build for the local CUDA environment.

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

Use the example configuration file as a starting point:

    configs/phase2_strict_rsift_random500k.example.yaml

Adapt the local dataset and output paths before running the pipeline.

Some scripts may also contain default path definitions that need to be adapted to the local machine before execution.

## Manifest format

The Phase 2 pipeline expects the ICDAR2017 Historical-WI TRAIN and TEST splits to follow the same writer-level protocol used in the thesis.

If local manifests are used, they should provide at least:

    path,writer_id,split

where `path` is relative to the corresponding TRAIN or TEST image directory.

## Execution

Run the programs from the `phase2_christlein2017/` folder.

Set the Python path:

    export PYTHONPATH="$PWD/src:$PYTHONPATH"

Then run the stages in order:

    python src/christlein2017_faithful/01r_extract_rsift_rootsift_patches.py
    python src/christlein2017_faithful/02r_fit_rsift_pca_kmeans_random500k.py
    python src/christlein2017_faithful/03r_build_surrogate_patch_dataset_rsift_random500k.py

    python scripts/christlein2017_faithful/04r_resnet20_full_surrogate_training_rsift_random500k.py
    python scripts/christlein2017_faithful/05r_extract_resnet20_embeddings_rsift_random500k.py

    python scripts/christlein2017_faithful/06r_mvlad_fit_encode_resnet20_rsift_random500k.py
    python scripts/christlein2017_faithful/06r_mvlad_encode_resnet20_rsift_random500k_global_ssr_l2.py
    python scripts/christlein2017_faithful/06r_mvlad_pca_whiten_resnet20_rsift_random500k.py
    python scripts/christlein2017_faithful/06r_retrieval_eval_resnet20_rsift_random500k_pca640.py

    python scripts/christlein2017_faithful/07r_exact_esvm_fe_resnet20_rsift_random500k_pca640.py

## Generated outputs

A full execution generates large intermediate artefacts, including:

- extracted RootSIFT descriptors
- extracted 32 x 32 patches
- RootSIFT PCA32 model
- KMeans surrogate model
- surrogate patch dataset and split files
- ResNet-20 checkpoints
- local learned embeddings
- m-VLAD codebooks
- page-level m-VLAD descriptors
- PCA640 model
- direct retrieval results
- E-SVM-FE features and final retrieval results
- runtime logs and result summaries

These generated files are not included in the GitHub repository.

## Notes

This folder contains code and lightweight configuration files only. It does not include Historical-WI images, extracted descriptors, extracted patches, learned embeddings, trained checkpoints, intermediate outputs, logs or result artefacts.

The SIFT stage should be described as **R-SIFT-like**, because it uses OpenCV SIFT followed by boundary, ink-content and duplicate-location filtering.

The implementation should be described as a strict re-implementation of the published methodology of Christlein et al., but not as a bitwise or source-code-level reproduction of the original author implementation.

## References

V. Christlein, M. Gropp, S. Fiel, and A. Maier, “Unsupervised Feature Learning for Writer Identification and Writer Retrieval,” ICDAR, 2017.

S. Fiel et al., “ICDAR2017 Competition on Historical Document Writer Identification (Historical-WI),” ICDAR, 2017.
