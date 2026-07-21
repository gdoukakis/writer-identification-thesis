"""
Phase 1 (Barcelona / CVC SRS-LBP) — ICDAR2017 Historical-WI protocol run.

Protocol (must be respected):
- Extract SRS-LBP features for TRAIN (1182) and TEST (3600) with identical preprocessing.
- Fit PCA ONLY on TRAIN features.
- Transform TRAIN and TEST using the fitted PCA.
- Apply power normalization + L2 normalization (as in the Barcelona-style pipeline).
- Retrieval evaluation: WITHIN-TEST retrieval on the 3600-image test set (exclude self-match).
  Rationale: TRAIN/TEST writers are disjoint (overlap=0), so TEST->TRAIN retrieval has no positives.

Outputs:
- NPY files for raw features and final embeddings (optional but useful for reproducibility)
- JSON log with parameters, timings, PCA stats, and metrics (thesis-ready)

Notes:
- This script assumes you already have:
- srs_lbp.py (srs_lbp_histogram_embedding)
- pca_and_normalize.py (PCANormalizer)
- evaluation.py (evaluate_retrieval_within_set and pairwise_squared_euclidean)
"""

from __future__ import annotations

import os
import time
import json
import platform
from dataclasses import asdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm

from srs_lbp import srs_lbp_histogram_embedding
from pca_and_normalize import PCANormalizer
from evaluation import evaluate_retrieval_within_set


# --------------------------------------------------------------------------------------
# I/O + preprocessing (keep identical for train/test)
# --------------------------------------------------------------------------------------

def load_grayscale_float32(path: str) -> np.ndarray:
    """Load an image, convert to grayscale, return float32 array."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def extract_single_embedding(
    img_path: str,
    discard_zero_mode: str = "keep_256_zeroed",
) -> np.ndarray:
    """Compute SRS-LBP embedding for one image (stateless, per-image)."""
    image = load_grayscale_float32(img_path)

    emb = srs_lbp_histogram_embedding(
        image=image,
        radii=range(1, 13),              # R = 1..12
        points=8,                        # P = 8
        discard_zero_mode=discard_zero_mode,
        l1_normalize_blocks=True,        # L1 per radius block
        valid_border=True                # consistent border policy
    )
    return emb.astype(np.float32, copy=False)

def extract_embeddings_parallel_checkpointed(
    df: pd.DataFrame,
    base_dir: str,
    discard_zero_mode: str,
    n_jobs: int,
    batch_size: int,
    desc: str,
    ckpt_dir: str,
) -> np.ndarray:
    """
    Parallel extraction with block-level checkpointing (resume-safe).

    - Saves each block as: ckpt_dir / block_{block_idx:05d}.npy
    - On restart, it loads existing blocks and continues from the first missing block.
    - At the end, it concatenates all blocks in order and returns X.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    img_paths = [os.path.join(base_dir, p) for p in df["path"]]
    n = len(img_paths)

    block_size = max(1, 8 * n_jobs)
    n_blocks = (n + block_size - 1) // block_size

    # Identify already completed blocks
    done = set()
    for fn in os.listdir(ckpt_dir):
        if fn.startswith("block_") and fn.endswith(".npy"):
            try:
                idx = int(fn.replace("block_", "").replace(".npy", ""))
                done.add(idx)
            except ValueError:
                pass

    print(f"{desc}: {n} images | block_size={block_size} | blocks={n_blocks} | done={len(done)}")

    # Extract missing blocks
    for block_idx in tqdm(range(n_blocks), desc=f"{desc} (resume)"):
        out_path = os.path.join(ckpt_dir, f"block_{block_idx:05d}.npy")
        if block_idx in done and os.path.exists(out_path):
            continue

        start = block_idx * block_size
        end = min(n, start + block_size)
        block = img_paths[start:end]

        block_embeddings = Parallel(n_jobs=n_jobs, backend="loky", batch_size=batch_size)(
            delayed(extract_single_embedding)(p, discard_zero_mode)
            for p in block
        )

        X_block = np.vstack(block_embeddings).astype(np.float32, copy=False)
        np.save(out_path, X_block)

    # Concatenate blocks in order
    blocks: List[np.ndarray] = []
    for block_idx in range(n_blocks):
        out_path = os.path.join(ckpt_dir, f"block_{block_idx:05d}.npy")
        if not os.path.exists(out_path):
            raise RuntimeError(f"Missing checkpoint block: {out_path}")
        blocks.append(np.load(out_path))

    X = np.vstack(blocks)
    if X.shape[0] != n:
        raise RuntimeError(f"Row count mismatch: got {X.shape[0]} rows, expected {n}")

    return X
    
# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------

def get_pca_stats(pca_norm: PCANormalizer) -> Dict[str, Any]:
    """Extract PCA stats for logging (thesis reproducibility)."""
    if pca_norm.pca_ is None:
        return {}

    pca = pca_norm.pca_
    evr = pca.explained_variance_ratio_
    cum = float(np.sum(evr)) if evr is not None else None

    return {
        "n_components": int(pca.n_components_),
        "input_dim": int(pca.n_features_in_),
        "whiten": bool(pca.whiten),
        "random_state": pca_norm.random_state,
        "explained_variance_ratio_first10": [float(x) for x in evr[:10]],
        "explained_variance_ratio_sum": cum,
    }


def write_text_summary(log_path_txt: str, log: Dict[str, Any]) -> None:
    """Write a human-readable TXT summary for thesis appendix."""
    lines = []
    lines.append("Phase 1 — Barcelona / CVC SRS-LBP (ICDAR2017 Historical-WI)")
    lines.append("Protocol: PCA fit on TRAIN (1182), retrieval WITHIN-TEST on TEST (3600), exclude self-match")
    lines.append("Note: TRAIN/TEST writers are disjoint => TEST->TRAIN retrieval has no positives (overlap=0).")
    lines.append("")

    cfg = log["config"]
    lines.append("=== Config ===")
    for k, v in cfg.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("=== Dataset ===")
    ds = log["dataset"]
    for k, v in ds.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("=== Timings (sec) ===")
    tm = log["timing_sec"]
    for k, v in tm.items():
        lines.append(f"{k}: {v:.4f}")
    lines.append("")

    lines.append("=== PCA Stats ===")
    ps = log["pca_stats"]
    for k, v in ps.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("=== Metrics (WITHIN-TEST) ===")
    mt = log["metrics"]
    lines.append(f"Top-1 accuracy: {mt['top1']:.6f}")
    lines.append(f"mAP:           {mt['mAP']:.6f}")
    lines.append("")

    with open(log_path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    # ----------------------------
    # User configuration (EDIT)
    # ----------------------------
    
    # Run this script from the repository root:
    # python src/icdar2017_srs_lbp_binarized_10022026.py

    # The manifest files are included in the GitHub repository.
    TRAIN_MANIFEST = r"manifests/manifest_train_1182.csv"
    TEST_MANIFEST = r"manifests/manifest_test_3600.csv"

    # Replace the following two values with the local directories
    # containing the binarised Historical-WI TRAIN and TEST images.
    TRAIN_BASE_DIR = r"REPLACE_WITH_PATH_TO_TRAIN_BINARISED_IMAGES"
    TEST_BASE_DIR = r"REPLACE_WITH_PATH_TO_TEST_BINARISED_IMAGES"

    # Generated descriptors, embeddings and logs will be written here.
    # This directory is created automatically and should not be uploaded to GitHub.
    OUTPUT_DIR = r"outputs/phase1_srs_lbp_binarized"

    # Extraction params
    DISCARD_ZERO_MODE = "keep_256_zeroed"
    N_JOBS = 4
    BATCH_SIZE = 1

    # PCA + normalization params
    N_COMPONENTS = 200
    WHITEN = False
    POWER_MODE = "signed_sqrt"   # "signed_sqrt" recommended for PCA outputs
    RANDOM_STATE = 0

    # Save intermediates (useful for reproducibility; set False if you want minimal outputs)
    SAVE_NPYS = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------
    # Load manifests
    # ----------------------------
    df_train = pd.read_csv(TRAIN_MANIFEST)
    df_test = pd.read_csv(TEST_MANIFEST)

    y_train = df_train["writer_id"].to_numpy()
    y_test = df_test["writer_id"].to_numpy()

    writer_overlap = set(y_train.tolist()).intersection(set(y_test.tolist()))
    overlap_size = int(len(writer_overlap))

    # Basic dataset sanity logs
    dataset_info = {
        "train_num_images": int(len(df_train)),
        "train_num_writers": int(df_train["writer_id"].nunique()),
        "train_images_per_writer_min": int(df_train["writer_id"].value_counts().min()),
        "train_images_per_writer_max": int(df_train["writer_id"].value_counts().max()),
        "test_num_images": int(len(df_test)),
        "test_num_writers": int(df_test["writer_id"].nunique()),
        "test_images_per_writer_min": int(df_test["writer_id"].value_counts().min()),
        "test_images_per_writer_max": int(df_test["writer_id"].value_counts().max()),
    }

    print("=== TRAIN ===")
    print(dataset_info["train_num_images"], "images |", dataset_info["train_num_writers"], "writers")
    print("=== TEST ===")
    print(dataset_info["test_num_images"], "images |", dataset_info["test_num_writers"], "writers")

    # ----------------------------
    # Feature extraction (SRS-LBP)
    # ----------------------------
    
    t0 = time.perf_counter()

    train_X_path = os.path.join(OUTPUT_DIR, "X_train_srs_lbp.npy")

    if os.path.exists(train_X_path):
        print("Loading existing TRAIN features...")
        X_train = np.load(train_X_path)
    else:
        print("Extracting TRAIN features...")
        train_ckpt = os.path.join(OUTPUT_DIR, "ckpt_train")
        X_train = extract_embeddings_parallel_checkpointed(
            df=df_train,
            base_dir=TRAIN_BASE_DIR,
            discard_zero_mode=DISCARD_ZERO_MODE,
            n_jobs=N_JOBS,
            batch_size=BATCH_SIZE,
            desc="Extract TRAIN",
            ckpt_dir=train_ckpt,
        )
        
    t1 = time.perf_counter()

    test_X_path = os.path.join(OUTPUT_DIR, "X_test_srs_lbp.npy")

    if os.path.exists(test_X_path):
        print("Loading existing TEST features...")
        X_test = np.load(test_X_path)
    else:
        print("Extracting TEST features...")
        test_ckpt = os.path.join(OUTPUT_DIR, "ckpt_test")
        X_test = extract_embeddings_parallel_checkpointed(
            df=df_test,
            base_dir=TEST_BASE_DIR,
            discard_zero_mode=DISCARD_ZERO_MODE,
            n_jobs=N_JOBS,
            batch_size=BATCH_SIZE,
            desc="Extract TEST",
            ckpt_dir=test_ckpt,
        )
    
    t2 = time.perf_counter()

    # Optional saves
    if SAVE_NPYS:        
        np.save(os.path.join(OUTPUT_DIR, "X_train_srs_lbp.npy"), X_train)
        np.save(os.path.join(OUTPUT_DIR, "y_train_writer_id.npy"), y_train)
        np.save(os.path.join(OUTPUT_DIR, "X_test_srs_lbp.npy"), X_test)
        np.save(os.path.join(OUTPUT_DIR, "y_test_writer_id.npy"), y_test)

    # ----------------------------
    # PCA fit on TRAIN only + transform TRAIN/TEST
    # ----------------------------
    pca_norm = PCANormalizer(
        n_components=N_COMPONENTS,
        whiten=WHITEN,
        power_mode=POWER_MODE,
        random_state=RANDOM_STATE
    )

    t3 = time.perf_counter()
    pca_norm.fit(X_train)                 # critical: TRAIN only
    Z_train = pca_norm.transform(X_train)
    Z_test = pca_norm.transform(X_test)
    t4 = time.perf_counter()

    if SAVE_NPYS:
        np.save(os.path.join(OUTPUT_DIR, "Z_train_pca_norm.npy"), Z_train)
        np.save(os.path.join(OUTPUT_DIR, "Z_test_pca_norm.npy"), Z_test)

    # ----------------------------
    # Retrieval evaluation: WITHIN-TEST (disjoint writers => TEST->TRAIN has no positives)
    # ----------------------------
    # IMPORTANT:
    # Retrieval is performed WITHIN the TEST set (query=gallery=test),
    # excluding self-match.
    # This follows the ICDAR2017 Historical-WI evaluation protocol,
    # given that TRAIN and TEST writers are disjoint (overlap=0).
    t5 = time.perf_counter()
    metrics = evaluate_retrieval_within_set(Z_test, y_test)
    t6 = time.perf_counter()
    
    print("=== Metrics (WITHIN-TEST) ===")
    print("Top-1 accuracy:", metrics.top1)
    print("mAP:", metrics.mAP)

    # ----------------------------
    # Logs for thesis
    # ----------------------------
    log: Dict[str, Any] = {
        "run_info": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataset": dataset_info,
        "config": {
            "train_manifest": TRAIN_MANIFEST,
            "test_manifest": TEST_MANIFEST,
            # "base_dir": BASE_DIR,
            "train_base_dir": TRAIN_BASE_DIR,
            "test_base_dir": TEST_BASE_DIR,
            "discard_zero_mode": DISCARD_ZERO_MODE,
            "radii": "1..12",
            "points": 8,
            "valid_border": True,
            "l1_normalize_blocks": True,
            "pca_n_components": N_COMPONENTS,
            "pca_whiten": WHITEN,
            "power_mode": POWER_MODE,
            "l2_normalize": True,
            "distance": "squared_euclidean",
            "n_jobs": N_JOBS,
            "batch_size": BATCH_SIZE,
            "save_npys": SAVE_NPYS,
            "evaluation_mode": "within_test",
            "train_test_writer_overlap": overlap_size,
        },
        "timing_sec": {
            "feature_extraction_train": float(t1 - t0),
            "feature_extraction_test": float(t2 - t1),
            "pca_fit_transform_total": float(t4 - t3),
            "retrieval_eval": float(t6 - t5),
            "total": float(t6 - t0),
        },
        "pca_stats": get_pca_stats(pca_norm),
        "metrics": {
            "top1": float(metrics.top1),
            "mAP": float(metrics.mAP),
        }
    }

    log_path_json = os.path.join(OUTPUT_DIR, "phase1_run_log.json")
    with open(log_path_json, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    log_path_txt = os.path.join(OUTPUT_DIR, "phase1_run_summary.txt")
    write_text_summary(log_path_txt, log)

    print("Saved log JSON:", log_path_json)
    print("Saved summary TXT:", log_path_txt)


if __name__ == "__main__":
    main()
