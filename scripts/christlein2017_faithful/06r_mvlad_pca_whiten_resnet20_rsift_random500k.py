"""
Stage 06R-B: PCA whitening for strict-branch ResNet-20 global SSR+L2 m-VLAD page descriptors.

Input:
- Stage 06R-N global SSR+L2 m-VLAD TRAIN/TEST matrices.

Output:
- PCA-whitened and L2-normalized TRAIN/TEST descriptors.
- PCA model and metadata.

Methodological status:
- PCA/whitening is fitted on TRAIN descriptors only.
- TEST descriptors are transformed using the TRAIN-fitted PCA model.
- Final descriptors are L2-normalized.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.decomposition import PCA


PROJECT_ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mvlad-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/mvlad_resnet20/resnet20_64d_mvlad_5xk64_global_ssr_l2_raw",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/mvlad_resnet20",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="resnet20_64d_mvlad_5xk64_global_ssr_l2_pca640",
    )

    parser.add_argument("--pca-dim", type=int, default=640)
    parser.add_argument("--whiten", action="store_true", default=True)
    parser.add_argument("--no-whiten", dest="whiten", action="store_false")
    parser.add_argument("--pre-pca-l2", action="store_true", default=True)
    parser.add_argument("--no-pre-pca-l2", dest="pre_pca_l2", action="store_false")
    parser.add_argument("--final-l2", action="store_true", default=True)
    parser.add_argument("--no-final-l2", dest="final_l2", action="store_false")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=123)

    return parser.parse_args()


def resolve_path(path_like: Any) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def l2_normalize_rows(x: np.ndarray, eps: float) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32, copy=False)


def matrix_stats(name: str, x: np.ndarray) -> Dict[str, Any]:
    norms = np.linalg.norm(x, axis=1)

    return {
        "name": name,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": bool(np.isfinite(x).all()),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
    }


def main() -> None:
    args = parse_args()

    mvlad_root = resolve_path(args.mvlad_root)
    out_root = resolve_path(args.output_root) / args.run_name
    models_dir = out_root / "models"
    final_dir = out_root / "final_descriptors"
    arrays_dir = out_root / "arrays"

    models_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    train_raw_path = mvlad_root / "arrays" / "train_raw_mvlad.npy"
    test_raw_path = mvlad_root / "arrays" / "test_raw_mvlad.npy"

    train_page_ids_path = mvlad_root / "arrays" / "train_page_ids.npy"
    test_page_ids_path = mvlad_root / "arrays" / "test_page_ids.npy"

    train_writer_ids_path = mvlad_root / "arrays" / "train_writer_ids.npy"
    test_writer_ids_path = mvlad_root / "arrays" / "test_writer_ids.npy"

    train_image_paths_path = mvlad_root / "arrays" / "train_image_paths.npy"
    test_image_paths_path = mvlad_root / "arrays" / "test_image_paths.npy"

    print("[LOAD]")
    print("train_raw:", train_raw_path)
    print("test_raw :", test_raw_path)

    train_raw = np.load(train_raw_path).astype(np.float32, copy=False)
    test_raw = np.load(test_raw_path).astype(np.float32, copy=False)

    if train_raw.ndim != 2 or test_raw.ndim != 2:
        raise RuntimeError(f"Expected 2D matrices, got train={train_raw.shape}, test={test_raw.shape}")

    if train_raw.shape[1] != test_raw.shape[1]:
        raise RuntimeError(f"Dim mismatch: train={train_raw.shape}, test={test_raw.shape}")

    if int(args.pca_dim) > min(train_raw.shape):
        raise RuntimeError(
            f"pca_dim={args.pca_dim} cannot exceed min(train_raw.shape)={min(train_raw.shape)}"
        )

    if not np.isfinite(train_raw).all():
        raise RuntimeError("Non-finite values in train_raw")
    if not np.isfinite(test_raw).all():
        raise RuntimeError("Non-finite values in test_raw")

    print("[RAW STATS]")
    print(json.dumps(matrix_stats("train_raw", train_raw), indent=2))
    print(json.dumps(matrix_stats("test_raw", test_raw), indent=2))

    train_input = train_raw
    test_input = test_raw

    if args.pre_pca_l2:
        print("[PRE PCA L2]")
        train_input = l2_normalize_rows(train_input, eps=float(args.eps))
        test_input = l2_normalize_rows(test_input, eps=float(args.eps))

    print("[FIT PCA]")
    print("pca_dim:", int(args.pca_dim))
    print("whiten:", bool(args.whiten))
    print("fit data:", train_input.shape)

    pca = PCA(
        n_components=int(args.pca_dim),
        whiten=bool(args.whiten),
        svd_solver="full",
        random_state=int(args.seed),
    )

    train_pca = pca.fit_transform(train_input).astype(np.float32, copy=False)
    test_pca = pca.transform(test_input).astype(np.float32, copy=False)

    if args.final_l2:
        print("[FINAL L2]")
        train_final = l2_normalize_rows(train_pca, eps=float(args.eps))
        test_final = l2_normalize_rows(test_pca, eps=float(args.eps))
    else:
        train_final = train_pca
        test_final = test_pca

    if not np.isfinite(train_final).all():
        raise RuntimeError("Non-finite values in train_final")
    if not np.isfinite(test_final).all():
        raise RuntimeError("Non-finite values in test_final")

    train_out = final_dir / f"train_pca{int(args.pca_dim)}_whiten_l2.npy"
    test_out = final_dir / f"test_pca{int(args.pca_dim)}_whiten_l2.npy"

    print("[SAVE FINAL]")
    print("train:", train_out, train_final.shape)
    print("test :", test_out, test_final.shape)

    np.save(train_out, train_final)
    np.save(test_out, test_final)

    # Copy metadata arrays to this run folder.
    metadata_pairs = [
        (train_page_ids_path, arrays_dir / "train_page_ids.npy"),
        (test_page_ids_path, arrays_dir / "test_page_ids.npy"),
        (train_writer_ids_path, arrays_dir / "train_writer_ids.npy"),
        (test_writer_ids_path, arrays_dir / "test_writer_ids.npy"),
        (train_image_paths_path, arrays_dir / "train_image_paths.npy"),
        (test_image_paths_path, arrays_dir / "test_image_paths.npy"),
    ]

    for src, dst in metadata_pairs:
        arr = np.load(src, allow_pickle=True)
        np.save(dst, arr)

    pca_model_path = models_dir / f"pca{int(args.pca_dim)}_whiten.pkl"
    save_pickle(pca_model_path, pca)

    explained = pca.explained_variance_ratio_
    singular_values = pca.singular_values_

    summary = {
        "stage": "06R_B_mvlad_pca_whiten_resnet20_rsift_random500k",
        "methodological_status": "strict_rsift_random500k_train_only_pca_whitening_global_ssr_l2_mvlad",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "input_descriptor_source": "Stage06R-N strict global SSR+L2 m-VLAD descriptors",
        "pca_fit_source": "TRAIN_ONLY_STAGE06R_N_GLOBAL_SSR_L2_MVLAD",
        "test_data_used_for_pca_fit": False,
        "global_normalization_before_pca": "signed_square_root_power_0.5_then_l2_after_mvlad_concat",
        "mvlad_root": str(mvlad_root),
        "output_root": str(out_root),
        "train_raw_path": str(train_raw_path),
        "test_raw_path": str(test_raw_path),
        "train_final_path": str(train_out),
        "test_final_path": str(test_out),
        "pca_model_path": str(pca_model_path),
        "raw_dim": int(train_raw.shape[1]),
        "pca_dim": int(args.pca_dim),
        "whiten": bool(args.whiten),
        "pre_pca_l2": bool(args.pre_pca_l2),
        "final_l2": bool(args.final_l2),
        "train_shape": list(train_final.shape),
        "test_shape": list(test_final.shape),
        "pca_explained_variance_ratio_sum": float(np.sum(explained)),
        "pca_explained_variance_ratio_first10": [float(x) for x in explained[:10]],
        "pca_singular_values_first10": [float(x) for x in singular_values[:10]],
        "train_raw_stats": matrix_stats("train_raw", train_raw),
        "test_raw_stats": matrix_stats("test_raw", test_raw),
        "train_final_stats": matrix_stats("train_final", train_final),
        "test_final_stats": matrix_stats("test_final", test_final),
        "test_data_used_for_pca_fit": False,
        "passed": True,
    }

    write_json(out_root / "run_summary.json", summary)

    print("[SUMMARY]")
    print(json.dumps(summary, indent=2)[:5000])
    print("[OK] Stage 06R-B strict-branch ResNet-20 PCA whitening completed.")
    print("[OUT]", out_root)


if __name__ == "__main__":
    main()
