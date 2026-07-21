# Phase 2: Christlein-faithful Writer Identification Pipeline

This repository contains the final accepted Phase 2 implementation used in the MSc thesis **Writer Identification**.

Final branch:

```text
strict_christlein2017_rsift_random500k
```

The code is a faithful re-implementation of the Christlein et al. (2017) writer identification pipeline at method level. It does not claim bitwise or source-code-level identity with the original author implementation.

## Pipeline

1. R-SIFT-like / RootSIFT extraction
2. 32 x 32 patch extraction
3. RootSIFT PCA whitening from 128 to 32 dimensions
4. random sample of 500,000 TRAIN descriptors
5. KMeans with K = 5,000 surrogate visual classes
6. ratio threshold 0.9
7. ResNet-20 surrogate training
8. 64-dimensional learned local embeddings
9. five m-VLAD codebooks
10. 64 centres per codebook
11. 500,000 TRAIN embeddings per codebook
12. global signed square-root normalisation
13. global L2 normalisation
14. PCA whitening from 20,480 to 640 dimensions
15. direct cosine retrieval
16. exact Exemplar-SVM Feature Encoding
17. TEST-only leave-one-image-out retrieval with self-match exclusion

## Accepted thesis results

| Method | Top-1 | mAP |
|---|---:|---:|
| Direct cosine retrieval | 86.69% | 70.28% |
| Exact E-SVM-FE | 87.47% | 71.70% |

## Dataset

The dataset is not included.

Expected protocol for ICDAR2017 Historical-WI:

| Split | Images | Writers |
|---|---:|---:|
| TRAIN | 1,182 | 394 |
| TEST | 3,600 | 720 |

Place the binarised images locally and adapt paths in the scripts or in:

```text
configs/phase2_strict_rsift_random500k.example.yaml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install PyTorch separately according to your CUDA setup.

## Run order

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"

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
```

## Notes

The repository intentionally excludes datasets, descriptors, extracted patches, embeddings, checkpoints, trained models and output artefacts.

The SIFT stage is described as **R-SIFT-like**, because it uses OpenCV SIFT followed by boundary, ink-content and duplicate-location filtering.
