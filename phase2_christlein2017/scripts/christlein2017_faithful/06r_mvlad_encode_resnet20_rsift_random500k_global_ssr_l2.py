#!/usr/bin/env python3
"""
Stage 06R-N: Strict-branch m-VLAD global SSR+L2 encoding for ResNet-20 64-D embeddings.

This variant reuses the already fitted strict Stage 06R-A m-VLAD codebooks and applies
the normalization strategy:

Current Stage 06A:
- per-codebook signed square-root / power normalization
- per-codebook L2
- concatenate the 5 normalized VLAD blocks
- no intra-normalization

This Stage 06R-N:
- compute raw VLAD residual block for each codebook
- concatenate all raw VLAD blocks
- apply global signed square-root / power normalization
- apply global L2 normalization
- no intra-normalization

No SIFT, no surrogate KMeans, and no ResNet training are repeated.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import MiniBatchKMeans


PROJECT_ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedding-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/resnet20_embeddings/resnet20_64d_epoch022_best",
    )
    parser.add_argument(
        "--codebook-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/mvlad_resnet20/resnet20_64d_mvlad_5xk64_raw",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/mvlad_resnet20",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="resnet20_64d_mvlad_5xk64_global_ssr_l2_raw",
    )

    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--local-dim", type=int, default=64)
    parser.add_argument("--num-codebooks", type=int, default=5)
    parser.add_argument("--num-clusters", type=int, default=64)
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--overwrite-encodings", action="store_true")

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


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def to_scalar_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        value = arr.item()
    elif arr.size == 1:
        value = arr.reshape(-1)[0].item()

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def to_scalar_int_or_string(value: Any):
    s = to_scalar_string(value)
    try:
        return int(s)
    except Exception:
        return s


def read_embedding_file(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        emb = np.asarray(z["embeddings"], dtype=np.float32)
        page_id = to_scalar_string(z["page_id"]) if "page_id" in z.files else path.stem
        writer_id = to_scalar_int_or_string(z["writer_id"]) if "writer_id" in z.files else ""
        image_path = to_scalar_string(z["image_path"]) if "image_path" in z.files else ""

    return {
        "embeddings": emb,
        "page_id": page_id,
        "writer_id": writer_id,
        "image_path": image_path,
    }


def list_page_files(embedding_root: Path, split: str) -> List[Path]:
    page_dir = embedding_root / split / "pages"
    files = sorted(page_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No embedding page files found: {page_dir}")
    return files


def load_existing_codebooks(
    codebook_root: Path,
    num_codebooks: int,
    num_clusters: int,
    local_dim: int,
) -> List[MiniBatchKMeans]:
    codebooks: List[MiniBatchKMeans] = []
    models_dir = codebook_root / "models"

    for cb in range(num_codebooks):
        model_path = models_dir / f"codebook_{cb:02d}.pkl"
        centers_path = models_dir / f"codebook_{cb:02d}_centers.npy"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing codebook model: {model_path}")
        if not centers_path.exists():
            raise FileNotFoundError(f"Missing codebook centers: {centers_path}")

        model = load_pickle(model_path)
        centers = np.load(centers_path)

        expected_shape = (num_clusters, local_dim)
        if centers.shape != expected_shape:
            raise RuntimeError(
                f"Invalid centers shape for codebook {cb}: {centers.shape}, "
                f"expected {expected_shape}"
            )

        if not np.isfinite(centers).all():
            raise RuntimeError(f"Non-finite centers in codebook {cb}")

        codebooks.append(model)
        print(f"[CODEBOOK {cb}] loaded {model_path}")

    return codebooks


def signed_power_global_l2(vec: np.ndarray, power: float, eps: float) -> np.ndarray:
    out = np.sign(vec) * (np.abs(vec) ** power)
    norm = float(np.linalg.norm(out))
    if norm > eps:
        out = out / norm
    return out.astype(np.float32, copy=False)


def compute_raw_vlad_block(
    local_desc: np.ndarray,
    model: MiniBatchKMeans,
) -> np.ndarray:
    centers = model.cluster_centers_.astype(np.float32, copy=False)

    if local_desc.ndim != 2:
        raise ValueError(f"local_desc must be 2D, got {local_desc.shape}")

    if local_desc.shape[1] != centers.shape[1]:
        raise ValueError(
            f"Descriptor dim mismatch: local={local_desc.shape[1]}, centers={centers.shape[1]}"
        )

    k, d = centers.shape
    vlad = np.zeros((k, d), dtype=np.float32)

    if len(local_desc) == 0:
        return vlad.reshape(-1)

    labels = model.predict(local_desc)
    residuals = local_desc.astype(np.float32, copy=False) - centers[labels]
    np.add.at(vlad, labels, residuals)

    return vlad.reshape(-1).astype(np.float32, copy=False)


def compute_mvlad_global_ssr_l2(
    local_desc: np.ndarray,
    codebooks: List[MiniBatchKMeans],
    power: float,
    eps: float,
) -> np.ndarray:
    raw_blocks = []

    for model in codebooks:
        raw_blocks.append(compute_raw_vlad_block(local_desc=local_desc, model=model))

    raw_concat = np.concatenate(raw_blocks, axis=0).astype(np.float32, copy=False)
    normalized = signed_power_global_l2(raw_concat, power=power, eps=eps)

    return normalized


def encode_split(
    split: str,
    embedding_root: Path,
    out_root: Path,
    codebooks: List[MiniBatchKMeans],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    in_files = list_page_files(embedding_root, split)

    out_page_dir = out_root / split / "pages"
    out_page_dir.mkdir(parents=True, exist_ok=True)

    arrays_dir = out_root / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    mvlad_dim = int(args.num_codebooks) * int(args.num_clusters) * int(args.local_dim)

    page_results = []
    failed_pages = []
    total_local_embeddings = 0

    t0 = time.time()

    print(f"[ENCODE {split.upper()}]")
    print("num_pages:", len(in_files))
    print("mvlad_dim:", mvlad_dim)

    for i, in_file in enumerate(in_files, start=1):
        out_file = out_page_dir / in_file.name

        if out_file.exists() and not args.overwrite_encodings:
            try:
                with np.load(out_file, allow_pickle=True) as z:
                    desc = z["mvlad"]
                    n_local = int(z["num_local_embeddings"])
                    page_id = to_scalar_string(z["page_id"])
                    writer_id = to_scalar_int_or_string(z["writer_id"])
                    image_path = to_scalar_string(z["image_path"])

                if desc.shape == (mvlad_dim,):
                    page_results.append({
                        "page_file": str(in_file),
                        "output_file": str(out_file),
                        "page_id": page_id,
                        "writer_id": writer_id,
                        "image_path": image_path,
                        "num_local_embeddings": n_local,
                        "mvlad_dim": mvlad_dim,
                        "skipped": True,
                    })
                    total_local_embeddings += n_local

                    if i == 1 or i % 100 == 0:
                        print(f"[{split}] skip {i}/{len(in_files)}")
                    continue
            except Exception:
                pass

        try:
            info = read_embedding_file(in_file)
            local_desc = info["embeddings"]

            if local_desc.ndim != 2 or local_desc.shape[1] != int(args.local_dim):
                raise ValueError(f"Invalid local embedding shape in {in_file}: {local_desc.shape}")

            if not np.isfinite(local_desc).all():
                raise RuntimeError(f"Non-finite local descriptors in {in_file}")

            desc = compute_mvlad_global_ssr_l2(
                local_desc=local_desc,
                codebooks=codebooks,
                power=float(args.power),
                eps=float(args.eps),
            )

            if desc.shape != (mvlad_dim,):
                raise RuntimeError(f"Invalid m-VLAD shape: {desc.shape}")

            if not np.isfinite(desc).all():
                raise RuntimeError(f"Non-finite m-VLAD descriptor in {in_file}")

            tmp = out_file.with_name(out_file.name + ".tmp.npz")
            np.savez_compressed(
                tmp,
                mvlad=desc,
                page_id=np.array(info["page_id"]),
                writer_id=np.array(info["writer_id"]),
                image_path=np.array(info["image_path"]),
                source_embedding_npz=np.array(str(in_file)),
                split=np.array(split),
                num_local_embeddings=np.array(int(local_desc.shape[0]), dtype=np.int64),
                mvlad_dim=np.array(mvlad_dim, dtype=np.int64),
                num_codebooks=np.array(int(args.num_codebooks), dtype=np.int64),
                num_clusters=np.array(int(args.num_clusters), dtype=np.int64),
                local_dim=np.array(int(args.local_dim), dtype=np.int64),
                normalization=np.array("global_ssr_power_0.5_l2_after_mvlad_concat_no_intra"),
            )
            os.replace(tmp, out_file)

            page_results.append({
                "page_file": str(in_file),
                "output_file": str(out_file),
                "page_id": info["page_id"],
                "writer_id": info["writer_id"],
                "image_path": info["image_path"],
                "num_local_embeddings": int(local_desc.shape[0]),
                "mvlad_dim": mvlad_dim,
                "skipped": False,
                "norm": float(np.linalg.norm(desc)),
            })
            total_local_embeddings += int(local_desc.shape[0])

            if i == 1 or i % 50 == 0:
                print(
                    f"[{split}] page={i}/{len(in_files)} "
                    f"local={local_desc.shape[0]} "
                    f"norm={np.linalg.norm(desc):.6f}"
                )

        except Exception as exc:
            failed = {
                "page_file": str(in_file),
                "error": repr(exc),
            }
            failed_pages.append(failed)
            print(f"[FAILED] {in_file}: {exc}")

    out_files = [out_page_dir / p.name for p in in_files]

    matrix = np.empty((len(out_files), mvlad_dim), dtype=np.float32)
    page_ids = []
    writer_ids = []
    image_paths = []
    local_counts = []

    for row_idx, p in enumerate(out_files):
        with np.load(p, allow_pickle=True) as z:
            matrix[row_idx] = z["mvlad"].astype(np.float32, copy=False)
            page_ids.append(to_scalar_string(z["page_id"]))
            writer_ids.append(to_scalar_int_or_string(z["writer_id"]))
            image_paths.append(to_scalar_string(z["image_path"]))
            local_counts.append(int(z["num_local_embeddings"]))

    matrix_path = arrays_dir / f"{split}_raw_mvlad.npy"
    page_ids_path = arrays_dir / f"{split}_page_ids.npy"
    writer_ids_path = arrays_dir / f"{split}_writer_ids.npy"
    image_paths_path = arrays_dir / f"{split}_image_paths.npy"
    local_counts_path = arrays_dir / f"{split}_local_counts.npy"

    np.save(matrix_path, matrix)
    np.save(page_ids_path, np.asarray(page_ids))
    np.save(writer_ids_path, np.asarray(writer_ids))
    np.save(image_paths_path, np.asarray(image_paths))
    np.save(local_counts_path, np.asarray(local_counts, dtype=np.int64))

    norms = np.linalg.norm(matrix, axis=1)

    summary = {
        "stage": "06R_N_mvlad_encode_resnet20_rsift_random500k_global_ssr_l2",
        "methodological_status": "strict_rsift_random500k_mvlad_global_ssr_l2_no_intra",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "embedding_source": "Stage05R strict ResNet-20 64-D embeddings",
        "codebook_source": "Stage06R-A TRAIN-only strict m-VLAD codebooks",
        "test_data_used_for_codebooks": False,
        "global_normalization": "signed_square_root_power_0.5_then_l2_after_mvlad_concat",
        "split": split,
        "num_pages": int(len(out_files)),
        "failed_pages": failed_pages,
        "num_failed_pages": int(len(failed_pages)),
        "total_local_embeddings": int(sum(local_counts)),
        "mvlad_dim": int(mvlad_dim),
        "num_codebooks": int(args.num_codebooks),
        "num_clusters": int(args.num_clusters),
        "local_dim": int(args.local_dim),
        "normalization": "global_ssr_power_0.5_l2_after_mvlad_concat_no_intra",
        "matrix_path": str(matrix_path),
        "page_ids_path": str(page_ids_path),
        "writer_ids_path": str(writer_ids_path),
        "image_paths_path": str(image_paths_path),
        "local_counts_path": str(local_counts_path),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "seconds": time.time() - t0,
        "passed": len(failed_pages) == 0,
    }

    write_json(out_root / split / f"{split}_mvlad_encoding_summary.json", summary)

    print(f"[ENCODE SUMMARY {split.upper()}]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


def main() -> None:
    args = parse_args()

    embedding_root = resolve_path(args.embedding_root)
    codebook_root = resolve_path(args.codebook_root)
    out_root = resolve_path(args.output_root) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    config = {
        "stage": "06R_N_mvlad_encode_resnet20_rsift_random500k_global_ssr_l2",
        "methodological_status": "strict_rsift_random500k_global_ssr_l2_reusing_strict_train_only_codebooks",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "embedding_source": "Stage05R strict ResNet-20 64-D embeddings",
        "codebook_source": "Stage06R-A TRAIN-only strict m-VLAD codebooks",
        "test_data_used_for_codebooks": False,
        "global_normalization": "signed_square_root_power_0.5_then_l2_after_mvlad_concat",
        "embedding_root": str(embedding_root),
        "codebook_root": str(codebook_root),
        "output_root": str(out_root),
        "splits": splits,
        "local_dim": int(args.local_dim),
        "num_codebooks": int(args.num_codebooks),
        "num_clusters": int(args.num_clusters),
        "power": float(args.power),
        "eps": float(args.eps),
        "normalization": "global_ssr_power_0.5_l2_after_mvlad_concat_no_intra",
        "codebooks_refit": False,
        "test_data_used_for_codebooks": False,
    }

    write_json(out_root / "run_config.json", config)

    codebooks = load_existing_codebooks(
        codebook_root=codebook_root,
        num_codebooks=int(args.num_codebooks),
        num_clusters=int(args.num_clusters),
        local_dim=int(args.local_dim),
    )

    split_summaries = []
    for split in splits:
        split_summaries.append(
            encode_split(
                split=split,
                embedding_root=embedding_root,
                out_root=out_root,
                codebooks=codebooks,
                args=args,
            )
        )

    run_summary = {
        "stage": "06R_N_mvlad_encode_resnet20_rsift_random500k_global_ssr_l2",
        "methodological_status": "strict_rsift_random500k_mvlad_global_ssr_l2_no_intra_reuse_strict_codebooks",
        "config": config,
        "split_summaries": split_summaries,
        "passed": all(s["passed"] for s in split_summaries),
    }

    write_json(out_root / "run_summary.json", run_summary)

    print("[OK] Stage 06R-N strict-branch m-VLAD global SSR+L2 encoding completed.")
    print("[OUT]", out_root)


if __name__ == "__main__":
    main()