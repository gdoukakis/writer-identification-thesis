"""
Stage 06R-A: Fit strict-branch m-VLAD codebooks and encode pages using ResNet-20 64-D embeddings.

Input:
- Stage 05R per-page strict-branch ResNet-20 64-D local embeddings.

Output:
- 5-codebook hard m-VLAD raw page descriptors.
- Per-page .npz files.
- Aggregate TRAIN/TEST .npy matrices and metadata arrays.

Methodological status:
- Strict-branch Christlein-compatible m-VLAD stage using ResNet-20 64-D descriptors.
- Codebooks are fitted on TRAIN local descriptors only.
- TEST descriptors are encoded using TRAIN-fitted codebooks.
- Hard VLAD assignment.
- Signed square-root power normalization with rho=0.5.
- L2 normalization per VLAD block.
- No intra-normalization.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        "--output-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/mvlad_resnet20",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="resnet20_64d_mvlad_5xk64_raw",
    )

    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--local-dim", type=int, default=64)
    parser.add_argument("--num-codebooks", type=int, default=5)
    parser.add_argument("--num-clusters", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--kmeans-batch-size", type=int, default=50000)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--kmeans-n-init", type=int, default=3)

    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-12)

    parser.add_argument("--overwrite-codebooks", action="store_true")
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


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
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


def scan_counts(page_files: List[Path], local_dim: int) -> Tuple[np.ndarray, int]:
    counts = []

    for i, p in enumerate(page_files, start=1):
        with np.load(p, allow_pickle=True) as z:
            emb = z["embeddings"]
            if emb.ndim != 2 or emb.shape[1] != local_dim:
                raise ValueError(f"Invalid embedding shape in {p}: {emb.shape}")
            counts.append(int(emb.shape[0]))

        if i == 1 or i % 250 == 0:
            print(f"[SCAN] {i}/{len(page_files)} pages")

    counts_arr = np.asarray(counts, dtype=np.int64)
    return counts_arr, int(counts_arr.sum())


def sample_train_descriptors(
    page_files: List[Path],
    counts: np.ndarray,
    total_descriptors: int,
    sample_size: int,
    local_dim: int,
    seed: int,
) -> np.ndarray:
    sample_n = min(int(sample_size), int(total_descriptors))
    rng = np.random.default_rng(seed)

    print(f"[SAMPLE] seed={seed} sample_n={sample_n} total={total_descriptors}")

    selected_global = np.sort(rng.choice(total_descriptors, size=sample_n, replace=False))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ends = starts + counts

    chunks = []
    selected_seen = 0

    for page_idx, p in enumerate(page_files):
        start = int(starts[page_idx])
        end = int(ends[page_idx])

        left = np.searchsorted(selected_global, start, side="left")
        right = np.searchsorted(selected_global, end, side="left")

        if right <= left:
            continue

        local_indices = selected_global[left:right] - start

        with np.load(p, allow_pickle=True) as z:
            emb = np.asarray(z["embeddings"], dtype=np.float32)
            chunk = emb[local_indices]

        chunks.append(chunk)
        selected_seen += len(chunk)

        if len(chunks) == 1 or len(chunks) % 100 == 0:
            print(f"[SAMPLE] chunks={len(chunks)} selected_seen={selected_seen}/{sample_n}")

    sample = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    if sample.shape != (sample_n, local_dim):
        raise RuntimeError(f"Unexpected sample shape: {sample.shape}, expected {(sample_n, local_dim)}")

    if not np.isfinite(sample).all():
        raise RuntimeError("Non-finite values detected in sampled TRAIN descriptors.")

    print("[SAMPLE] final shape:", sample.shape)
    return sample


def fit_or_load_codebooks(
    train_files: List[Path],
    train_counts: np.ndarray,
    total_train_descriptors: int,
    args: argparse.Namespace,
    out_root: Path,
) -> Tuple[List[MiniBatchKMeans], Dict[str, Any]]:
    model_dir = out_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    codebooks = []
    summaries = []

    for cb in range(int(args.num_codebooks)):
        model_path = model_dir / f"codebook_{cb:02d}.pkl"
        centers_path = model_dir / f"codebook_{cb:02d}_centers.npy"

        if model_path.exists() and centers_path.exists() and not args.overwrite_codebooks:
            print(f"[CODEBOOK {cb}] loading existing model: {model_path}")
            model = load_pickle(model_path)
            centers = np.load(centers_path)

            if centers.shape != (int(args.num_clusters), int(args.local_dim)):
                raise RuntimeError(f"Invalid centers shape for codebook {cb}: {centers.shape}")

            codebooks.append(model)
            summaries.append({
                "codebook": int(cb),
                "status": "loaded_existing",
                "model_path": str(model_path),
                "centers_path": str(centers_path),
                "centers_shape": list(centers.shape),
            })
            continue

        sample_seed = int(args.seed) + cb
        sample = sample_train_descriptors(
            page_files=train_files,
            counts=train_counts,
            total_descriptors=total_train_descriptors,
            sample_size=int(args.sample_size),
            local_dim=int(args.local_dim),
            seed=sample_seed,
        )

        print(f"[CODEBOOK {cb}] fitting MiniBatchKMeans")

        t0 = time.time()

        model = MiniBatchKMeans(
            n_clusters=int(args.num_clusters),
            batch_size=int(args.kmeans_batch_size),
            max_iter=int(args.kmeans_max_iter),
            n_init=int(args.kmeans_n_init),
            random_state=sample_seed,
            verbose=0,
            reassignment_ratio=0.01,
        )

        model.fit(sample)

        centers = model.cluster_centers_.astype(np.float32, copy=False)

        save_pickle(model_path, model)
        np.save(centers_path, centers)

        inertia = float(model.inertia_)
        elapsed = time.time() - t0

        print(f"[CODEBOOK {cb}] done seconds={elapsed:.2f} inertia={inertia:.4f}")

        codebooks.append(model)
        summaries.append({
            "codebook": int(cb),
            "status": "fit",
            "sample_seed": int(sample_seed),
            "sample_size": int(sample.shape[0]),
            "num_clusters": int(args.num_clusters),
            "local_dim": int(args.local_dim),
            "model_path": str(model_path),
            "centers_path": str(centers_path),
            "centers_shape": list(centers.shape),
            "inertia": inertia,
            "seconds": elapsed,
        })

    summary = {
        "stage": "06R_fit_mvlad_codebooks_resnet20_rsift_random500k",
        "methodological_status": "strict_rsift_random500k_train_only_codebook_fitting",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "embedding_source": "Stage05R strict ResNet-20 64-D embeddings",
        "codebook_fitting_source": "TRAIN_ONLY_STAGE05R_EMBEDDINGS",
        "sampling_strategy": "global_random_train_embeddings_per_codebook",
        "sample_source": "TRAIN_ONLY",
        "page_balanced_sampling": False,
        "test_data_used_for_codebooks": False,
        "num_codebooks": int(args.num_codebooks),
        "num_clusters": int(args.num_clusters),
        "local_dim": int(args.local_dim),
        "sample_size_per_codebook": int(args.sample_size),
        "total_train_descriptors": int(total_train_descriptors),
        "kmeans_batch_size": int(args.kmeans_batch_size),
        "kmeans_max_iter": int(args.kmeans_max_iter),
        "kmeans_n_init": int(args.kmeans_n_init),
        "seed": int(args.seed),
        "codebooks": summaries,
        "test_data_used": False,
        "passed": True,
    }

    write_json(out_root / "codebook_fit_summary.json", summary)
    return codebooks, summary


def signed_power_l2_normalize(vec: np.ndarray, power: float, eps: float) -> np.ndarray:
    out = np.sign(vec) * (np.abs(vec) ** power)
    norm = float(np.linalg.norm(out))
    if norm > eps:
        out = out / norm
    return out.astype(np.float32, copy=False)


def compute_vlad_block(
    local_desc: np.ndarray,
    centers: np.ndarray,
    model: MiniBatchKMeans,
    power: float,
    eps: float,
) -> np.ndarray:
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

    flat = vlad.reshape(-1)
    flat = signed_power_l2_normalize(flat, power=power, eps=eps)
    return flat


def compute_mvlad_descriptor(
    local_desc: np.ndarray,
    codebooks: List[MiniBatchKMeans],
    power: float,
    eps: float,
) -> np.ndarray:
    blocks = []

    for model in codebooks:
        centers = model.cluster_centers_.astype(np.float32, copy=False)
        block = compute_vlad_block(
            local_desc=local_desc,
            centers=centers,
            model=model,
            power=power,
            eps=eps,
        )
        blocks.append(block)

    return np.concatenate(blocks, axis=0).astype(np.float32, copy=False)


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

            if local_desc.shape[1] != int(args.local_dim):
                raise ValueError(f"Invalid local dim in {in_file}: {local_desc.shape}")

            if not np.isfinite(local_desc).all():
                raise RuntimeError(f"Non-finite local descriptors in {in_file}")

            desc = compute_mvlad_descriptor(
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
                normalization=np.array("per_codebook_ssr_power_0.5_l2_no_intra"),
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

    # Build aggregate arrays in sorted page order.
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
        "stage": "06R_mvlad_encode_resnet20_rsift_random500k",
        "methodological_status": "strict_rsift_random500k_resnet20_64d_mvlad_encoding_per_codebook_ssr_l2",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "embedding_source": "Stage05R strict ResNet-20 64-D embeddings",
        "codebook_source": "Stage06R-A TRAIN-only strict m-VLAD codebooks",
        "test_data_used_for_codebooks": False,
        "split": split,
        "num_pages": int(len(out_files)),
        "failed_pages": failed_pages,
        "num_failed_pages": int(len(failed_pages)),
        "total_local_embeddings": int(sum(local_counts)),
        "mvlad_dim": int(mvlad_dim),
        "num_codebooks": int(args.num_codebooks),
        "num_clusters": int(args.num_clusters),
        "local_dim": int(args.local_dim),
        "normalization": "per_codebook_ssr_power_0.5_l2_no_intra",
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
    out_root = resolve_path(args.output_root) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    train_files = list_page_files(embedding_root, "train")
    train_counts, total_train = scan_counts(train_files, int(args.local_dim))

    config = {
        "stage": "06R_mvlad_fit_encode_resnet20_rsift_random500k",
        "methodological_status": "strict_rsift_random500k_resnet20_64d_mvlad_fit_encode_train_only_codebooks",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "embedding_source": "Stage05R strict ResNet-20 64-D embeddings",
        "codebook_fitting_source": "TRAIN_ONLY_STAGE05R_EMBEDDINGS",
        "sampling_strategy": "global_random_train_embeddings_per_codebook",
        "sample_source": "TRAIN_ONLY",
        "page_balanced_sampling": False,
        "embedding_root": str(embedding_root),
        "output_root": str(out_root),
        "splits": splits,
        "local_dim": int(args.local_dim),
        "num_codebooks": int(args.num_codebooks),
        "num_clusters": int(args.num_clusters),
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "kmeans_batch_size": int(args.kmeans_batch_size),
        "kmeans_max_iter": int(args.kmeans_max_iter),
        "kmeans_n_init": int(args.kmeans_n_init),
        "power": float(args.power),
        "eps": float(args.eps),
        "normalization": "per_codebook_ssr_power_0.5_l2_no_intra",
        "test_data_used_for_codebooks": False,
    }

    write_json(out_root / "run_config.json", config)

    codebooks, fit_summary = fit_or_load_codebooks(
        train_files=train_files,
        train_counts=train_counts,
        total_train_descriptors=total_train,
        args=args,
        out_root=out_root,
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
        "stage": "06R_mvlad_fit_encode_resnet20_rsift_random500k",
        "methodological_status": "strict_rsift_random500k_resnet20_64d_mvlad_fit_encode_train_only_codebooks",
        "config": config,
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "embedding_source": "Stage05R strict ResNet-20 64-D embeddings",
        "codebook_fitting_source": "TRAIN_ONLY_STAGE05R_EMBEDDINGS",
        "sampling_strategy": "global_random_train_embeddings_per_codebook",
        "sample_source": "TRAIN_ONLY",
        "page_balanced_sampling": False,
        "codebook_fit_summary": fit_summary,
        "split_summaries": split_summaries,
        "passed": all(s["passed"] for s in split_summaries),
    }

    write_json(out_root / "run_summary.json", run_summary)

    print("[OK] Stage 06R-A strict-branch ResNet-20 m-VLAD fitting/encoding completed.")
    print("[OUT]", out_root)


if __name__ == "__main__":
    main()
