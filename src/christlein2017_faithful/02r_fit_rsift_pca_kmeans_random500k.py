from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


PROJECT_ROOT = Path(".")

DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "faithful_christlein2017"
    / "strict_christlein2017_rsift_random500k"
    / "rsift_rootsift_patches"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "faithful_christlein2017"
    / "strict_christlein2017_rsift_random500k"
    / "rsift_pca_kmeans_random500k"
)


@dataclass
class Stage02Config:
    """Configuration for SIFT PCA whitening and KMeans visual vocabulary fitting."""

    pca_dim: int = 32
    pca_whiten: bool = True

    sample_max_descriptors: int = 500_000
    sample_seed: int = 1

    transform_chunk_size: int = 250_000

    kmeans_clusters: int = 5000
    kmeans_batch_size: int = 50_000
    kmeans_max_iter: int = 100
    kmeans_n_init: int = 3


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Write a numpy array atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp_path, path)


def npy_is_valid(path: Path, expected_shape: Tuple[int, int], dtype: Any = np.float32) -> bool:
    """Check whether a .npy file exists, is readable, and has the expected shape."""
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == expected_shape and arr.dtype == dtype
    except Exception:
        return False


def load_split_paths(input_root: Path, split: str) -> Dict[str, Path]:
    """Return Stage 01 file paths for a split."""
    split_root = input_root / split
    return {
        "split_root": split_root,
        "rootsift": split_root / f"{split}_rootsift_descriptors.npy",
        "page_stats": split_root / f"{split}_page_stats.csv",
        "metadata": split_root / f"{split}_patch_metadata.csv",
    }


def load_rootsift_memmap(input_root: Path, split: str) -> np.ndarray:
    """Load RootSIFT descriptors as a memory-mapped array."""
    paths = load_split_paths(input_root, split)
    if not paths["rootsift"].exists():
        raise FileNotFoundError(f"Missing RootSIFT descriptor file: {paths['rootsift']}")
    arr = np.load(paths["rootsift"], mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 128:
        raise ValueError(f"Expected {split} RootSIFT shape [N, 128], got {arr.shape}")
    return arr


def load_page_stats(input_root: Path, split: str) -> pd.DataFrame:
    """Load page-level stats from Stage 01."""
    paths = load_split_paths(input_root, split)
    if not paths["page_stats"].exists():
        raise FileNotFoundError(f"Missing page stats file: {paths['page_stats']}")

    df = pd.read_csv(paths["page_stats"])

    required = {"image_id", "writer_id", "total_keypoints_kept"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{paths['page_stats']} missing columns: {sorted(missing)}")

    return df


def build_page_descriptor_ranges(stats: pd.DataFrame) -> pd.DataFrame:
    """Add descriptor start/end rows per page based on total_keypoints_kept."""
    out = stats.copy()
    counts = out["total_keypoints_kept"].astype(int).to_numpy()

    starts = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        starts[1:] = np.cumsum(counts[:-1])

    ends = starts + counts

    out["descriptor_start"] = starts
    out["descriptor_end"] = ends
    return out


def global_random_train_sample_indices(
    num_descriptors: int,
    sample_max: int,
    seed: int,
) -> np.ndarray:
    """Sample TRAIN descriptor indices uniformly at random from all TRAIN descriptors.

    Strict Christlein-style variant:
    - sample from the full TRAIN descriptor pool
    - no page balancing
    - no TEST descriptors
    - no replacement unless the pool is smaller than the requested sample
    """
    if sample_max <= 0:
        raise ValueError("sample_max must be positive")
    if num_descriptors <= 0:
        raise ValueError("num_descriptors must be positive")

    rng = np.random.default_rng(seed)

    if num_descriptors <= sample_max:
        indices = np.arange(num_descriptors, dtype=np.int64)
    else:
        indices = rng.choice(
            num_descriptors,
            size=sample_max,
            replace=False,
        ).astype(np.int64)

    rng.shuffle(indices)
    return indices


def sample_indices_path(output_root: Path) -> Path:
    """Path for saved global-random TRAIN descriptor sample indices."""
    return output_root / "samples" / "train_global_random_sample_indices_500k.npy"


def fit_pca(
    input_root: Path,
    output_root: Path,
    cfg: Stage02Config,
    overwrite: bool,
) -> Path:
    """Fit PCA whitening on TRAIN RootSIFT descriptors only."""
    output_root.mkdir(parents=True, exist_ok=True)

    pca_dir = output_root / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    pca_path = pca_dir / "sift_rootsift_pca_128_to_32_whiten.joblib"
    summary_path = pca_dir / "pca_fit_summary.json"
    sample_path = sample_indices_path(output_root)

    if pca_path.exists() and sample_path.exists() and not overwrite:
        print(f"[SKIP] PCA already exists: {pca_path}")
        return pca_path

    train_rs = load_rootsift_memmap(input_root, "train")
    train_stats = load_page_stats(input_root, "train")
    page_ranges = build_page_descriptor_ranges(train_stats)

    expected_train_rows = int(page_ranges["total_keypoints_kept"].sum())
    if expected_train_rows != train_rs.shape[0]:
        raise ValueError(
            f"TRAIN stats sum {expected_train_rows} != descriptor rows {train_rs.shape[0]}"
        )

    print("[PCA] Building global-random TRAIN sample indices...")
    indices = global_random_train_sample_indices(
        num_descriptors=int(train_rs.shape[0]),
        sample_max=cfg.sample_max_descriptors,
        seed=cfg.sample_seed,
    )

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(sample_path, indices)

    print(f"[PCA] Sample size: {len(indices)}")
    print("[PCA] Loading TRAIN RootSIFT sample into memory...")
    sample = np.asarray(train_rs[indices], dtype=np.float32)

    print("[PCA] Fitting PCA 128 -> 32 with whitening on TRAIN sample only...")
    pca = PCA(
        n_components=cfg.pca_dim,
        whiten=cfg.pca_whiten,
        svd_solver="full",
        random_state=cfg.sample_seed,
    )
    pca.fit(sample)

    joblib.dump(pca, pca_path)

    explained = pca.explained_variance_ratio_
    summary = {
        "stage": "fit_pca",
        "fit_split": "train",
        "test_used_for_fit": False,
        "sampling_strategy": "global_random_train_descriptors",
        "sample_source": "TRAIN_ONLY",
        "test_used_for_sampling": False,
        "page_balanced_sampling": False,
        "input_descriptor_shape": [int(train_rs.shape[0]), int(train_rs.shape[1])],
        "sample_size": int(len(indices)),
        "sample_indices_path": str(sample_path),
        "pca_path": str(pca_path),
        "pca_dim": int(cfg.pca_dim),
        "pca_whiten": bool(cfg.pca_whiten),
        "explained_variance_ratio_sum": float(explained.sum()),
        "explained_variance_ratio_first_10": [float(x) for x in explained[:10]],
        "config": asdict(cfg),
    }
    atomic_write_json(summary_path, summary)

    print(f"[OK] PCA saved to: {pca_path}")
    print(f"[OK] PCA summary saved to: {summary_path}")
    return pca_path


def chunk_file_path(chunks_dir: Path, split: str, start: int, end: int) -> Path:
    """Return chunk file path for a descriptor range."""
    return chunks_dir / f"{split}_pca32_{start:010d}_{end:010d}.npy"


def transform_split_to_chunks(
    input_root: Path,
    output_root: Path,
    split: str,
    cfg: Stage02Config,
    overwrite_chunks: bool,
) -> Dict[str, Any]:
    """Transform a split's RootSIFT descriptors to PCA-32 chunks."""
    pca_path = output_root / "pca" / "sift_rootsift_pca_128_to_32_whiten.joblib"
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA model not found: {pca_path}")

    pca = joblib.load(pca_path)
    descriptors = load_rootsift_memmap(input_root, split)

    split_out = output_root / split
    chunks_dir = split_out / "pca32_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    n = int(descriptors.shape[0])
    d_out = int(cfg.pca_dim)
    chunk_size = int(cfg.transform_chunk_size)

    completed = 0
    processed = 0
    chunk_records: List[Dict[str, Any]] = []

    for start in tqdm(range(0, n, chunk_size), desc=f"Transform {split}"):
        end = min(start + chunk_size, n)
        expected_shape = (end - start, d_out)
        out_path = chunk_file_path(chunks_dir, split, start, end)

        if npy_is_valid(out_path, expected_shape) and not overwrite_chunks:
            completed += 1
            chunk_records.append(
                {"start": int(start), "end": int(end), "path": str(out_path), "status": "skipped"}
            )
            continue

        x = np.asarray(descriptors[start:end], dtype=np.float32)
        y = pca.transform(x).astype(np.float32)

        if y.shape != expected_shape:
            raise ValueError(f"Unexpected PCA chunk shape {y.shape}, expected {expected_shape}")
        if not np.isfinite(y).all():
            raise ValueError(f"Non-finite values in PCA chunk {start}:{end}")

        atomic_save_npy(out_path, y)

        processed += 1
        chunk_records.append(
            {"start": int(start), "end": int(end), "path": str(out_path), "status": "processed"}
        )

    summary = {
        "stage": "transform_split_to_chunks",
        "split": split,
        "input_shape": [int(descriptors.shape[0]), int(descriptors.shape[1])],
        "output_dim": int(d_out),
        "chunk_size": int(chunk_size),
        "processed_chunks_this_run": int(processed),
        "skipped_chunks_this_run": int(completed),
        "num_chunks_total": int(len(chunk_records)),
        "chunks_dir": str(chunks_dir),
        "chunks": chunk_records,
    }

    atomic_write_json(split_out / f"{split}_pca32_transform_chunks_summary.json", summary)
    print(f"[OK] {split} PCA transform chunks complete")
    return summary


def aggregate_pca32_chunks(
    output_root: Path,
    split: str,
    cfg: Stage02Config,
    overwrite: bool,
) -> Path:
    """Aggregate PCA-32 chunk files into one split-level .npy memmap."""
    split_out = output_root / split
    chunks_summary_path = split_out / f"{split}_pca32_transform_chunks_summary.json"
    out_path = split_out / f"{split}_rootsift_pca32.npy"

    if out_path.exists() and not overwrite:
        arr = np.load(out_path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[1] == cfg.pca_dim:
            print(f"[SKIP] Aggregated PCA-32 already exists: {out_path}")
            return out_path

    if not chunks_summary_path.exists():
        raise FileNotFoundError(f"Missing chunks summary: {chunks_summary_path}")

    with chunks_summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    chunks = summary["chunks"]
    total_rows = int(summary["input_shape"][0])
    d_out = int(summary["output_dim"])

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    mmap = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, d_out),
    )

    for rec in tqdm(chunks, desc=f"Aggregate PCA-32 {split}"):
        start = int(rec["start"])
        end = int(rec["end"])
        chunk_path = Path(rec["path"])

        if not npy_is_valid(chunk_path, (end - start, d_out)):
            raise FileNotFoundError(f"Missing or invalid chunk: {chunk_path}")

        chunk = np.load(chunk_path, mmap_mode="r")
        mmap[start:end] = chunk

    mmap.flush()
    del mmap

    os.replace(tmp_path, out_path)

    aggregate_summary = {
        "stage": "aggregate_pca32_chunks",
        "split": split,
        "output_path": str(out_path),
        "output_shape": [int(total_rows), int(d_out)],
        "num_chunks": int(len(chunks)),
    }
    atomic_write_json(split_out / f"{split}_pca32_aggregate_summary.json", aggregate_summary)

    print(f"[OK] Aggregated PCA-32 saved to: {out_path}")
    return out_path


def fit_kmeans(
    output_root: Path,
    cfg: Stage02Config,
    overwrite: bool,
) -> Path:
    """Fit MiniBatchKMeans K=5000 on TRAIN PCA-32 descriptors only."""
    kmeans_dir = output_root / "kmeans"
    kmeans_dir.mkdir(parents=True, exist_ok=True)

    train_pca_path = output_root / "train" / "train_rootsift_pca32.npy"
    sample_path = sample_indices_path(output_root)

    kmeans_tag = f"k{cfg.kmeans_clusters}"
    kmeans_path = kmeans_dir / f"minibatch_kmeans_{kmeans_tag}_train_pca32.joblib"
    centers_path = kmeans_dir / f"minibatch_kmeans_{kmeans_tag}_centers.npy"
    summary_path = kmeans_dir / f"kmeans_fit_summary_{kmeans_tag}.json"

    if kmeans_path.exists() and centers_path.exists() and not overwrite:
        print(f"[SKIP] KMeans already exists: {kmeans_path}")
        return kmeans_path

    if not train_pca_path.exists():
        raise FileNotFoundError(f"Missing TRAIN PCA-32 descriptors: {train_pca_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing TRAIN sample indices: {sample_path}")

    train_pca = np.load(train_pca_path, mmap_mode="r")
    sample_indices = np.load(sample_path, mmap_mode="r")

    if train_pca.ndim != 2 or train_pca.shape[1] != cfg.pca_dim:
        raise ValueError(f"Unexpected TRAIN PCA shape: {train_pca.shape}")

    print("[KMEANS] Loading sampled TRAIN PCA-32 descriptors...")
    sample = np.asarray(train_pca[sample_indices], dtype=np.float32)

    if sample.shape[0] < cfg.kmeans_clusters:
        raise ValueError(
            f"KMeans sample size {sample.shape[0]} is smaller than clusters {cfg.kmeans_clusters}"
        )

    print(
        f"[KMEANS] Fitting MiniBatchKMeans K={cfg.kmeans_clusters}, "
        f"batch_size={cfg.kmeans_batch_size}, sample={sample.shape}"
    )

    kmeans = MiniBatchKMeans(
        n_clusters=cfg.kmeans_clusters,
        batch_size=cfg.kmeans_batch_size,
        max_iter=cfg.kmeans_max_iter,
        n_init=cfg.kmeans_n_init,
        compute_labels=False,
        random_state=cfg.sample_seed,
        verbose=0,
    )
    kmeans.fit(sample)

    centers = kmeans.cluster_centers_.astype(np.float32)

    joblib.dump(kmeans, kmeans_path)
    atomic_save_npy(centers_path, centers)

    summary = {
        "stage": "fit_kmeans",
        "fit_split": "train",
        "test_used_for_fit": False,
        "sampling_strategy": "global_random_train_descriptors",
        "sample_source": "TRAIN_ONLY",
        "test_used_for_sampling": False,
        "page_balanced_sampling": False,
        "train_pca_path": str(train_pca_path),
        "sample_indices_path": str(sample_path),
        "sample_shape": [int(sample.shape[0]), int(sample.shape[1])],
        "kmeans_path": str(kmeans_path),
        "centers_path": str(centers_path),
        "centers_shape": [int(centers.shape[0]), int(centers.shape[1])],
        "kmeans_clusters": int(cfg.kmeans_clusters),
        "kmeans_batch_size": int(cfg.kmeans_batch_size),
        "kmeans_max_iter": int(cfg.kmeans_max_iter),
        "kmeans_n_init": int(cfg.kmeans_n_init),
        "inertia": float(kmeans.inertia_) if hasattr(kmeans, "inertia_") else None,
        "n_iter": int(kmeans.n_iter_) if hasattr(kmeans, "n_iter_") else None,
        "config": asdict(cfg),
    }
    atomic_write_json(summary_path, summary)

    print(f"[OK] KMeans saved to: {kmeans_path}")
    print(f"[OK] centers saved to: {centers_path}")
    return kmeans_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 02: TRAIN-only SIFT PCA whitening 128->32 and KMeans K=5000."
    )

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "fit_pca",
            "transform_train",
            "transform_test",
            "aggregate_train",
            "aggregate_test",
            "fit_kmeans",
        ],
        default="all",
    )

    parser.add_argument("--input-root", type=str, default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))

    parser.add_argument("--sample-max-descriptors", type=int, default=500_000)
    parser.add_argument("--sample-seed", type=int, default=1)

    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--transform-chunk-size", type=int, default=250_000)

    parser.add_argument("--kmeans-clusters", type=int, default=5000)
    parser.add_argument("--kmeans-batch-size", type=int, default=50_000)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--kmeans-n-init", type=int, default=3)

    parser.add_argument("--overwrite-pca", action="store_true")
    parser.add_argument("--overwrite-chunks", action="store_true")
    parser.add_argument("--overwrite-aggregate", action="store_true")
    parser.add_argument("--overwrite-kmeans", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Stage02Config(
        pca_dim=int(args.pca_dim),
        pca_whiten=True,
        sample_max_descriptors=int(args.sample_max_descriptors),
        sample_seed=int(args.sample_seed),
        transform_chunk_size=int(args.transform_chunk_size),
        kmeans_clusters=int(args.kmeans_clusters),
        kmeans_batch_size=int(args.kmeans_batch_size),
        kmeans_max_iter=int(args.kmeans_max_iter),
        kmeans_n_init=int(args.kmeans_n_init),
    )

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("[CONFIG]")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    print(f"[INPUT_ROOT] {input_root}")
    print(f"[OUTPUT_ROOT] {output_root}")
    print(f"[MODE] {args.mode}")

    if args.mode in ["all", "fit_pca"]:
        fit_pca(
            input_root=input_root,
            output_root=output_root,
            cfg=cfg,
            overwrite=bool(args.overwrite_pca),
        )

    if args.mode in ["all", "transform_train"]:
        transform_split_to_chunks(
            input_root=input_root,
            output_root=output_root,
            split="train",
            cfg=cfg,
            overwrite_chunks=bool(args.overwrite_chunks),
        )

    if args.mode in ["all", "aggregate_train"]:
        aggregate_pca32_chunks(
            output_root=output_root,
            split="train",
            cfg=cfg,
            overwrite=bool(args.overwrite_aggregate),
        )

    if args.mode in ["all", "transform_test"]:
        transform_split_to_chunks(
            input_root=input_root,
            output_root=output_root,
            split="test",
            cfg=cfg,
            overwrite_chunks=bool(args.overwrite_chunks),
        )

    if args.mode in ["all", "aggregate_test"]:
        aggregate_pca32_chunks(
            output_root=output_root,
            split="test",
            cfg=cfg,
            overwrite=bool(args.overwrite_aggregate),
        )

    if args.mode in ["all", "fit_kmeans"]:
        fit_kmeans(
            output_root=output_root,
            cfg=cfg,
            overwrite=bool(args.overwrite_kmeans),
        )

    stage_summary = {
        "stage": "02_fit_sift_pca_kmeans",
        "mode": args.mode,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "config": asdict(cfg),
        "methodological_notes": {
            "pca_fit_split": "train_only",
            "kmeans_fit_split": "train_only",
            "test_used_for_fitting": False,
            "pca_dim": "128_to_32",
            "pca_whitening": True,
            "kmeans_clusters": cfg.kmeans_clusters,
            "kmeans_max_train_descriptors": cfg.sample_max_descriptors,
        },
    }
    atomic_write_json(output_root / "stage02_last_run_summary.json", stage_summary)

    print("[OK] Stage 02 mode completed")


if __name__ == "__main__":
    main()