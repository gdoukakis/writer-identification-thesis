from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(".")

DEFAULT_STAGE01_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "faithful_christlein2017"
    / "strict_christlein2017_rsift_random500k"
    / "rsift_rootsift_patches"
)

DEFAULT_STAGE02_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "faithful_christlein2017"
    / "strict_christlein2017_rsift_random500k"
    / "rsift_pca_kmeans_random500k"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "faithful_christlein2017"
    / "strict_christlein2017_rsift_random500k"
    / "surrogate_assignments_k5000_ratio09"
)


@dataclass
class Stage03Config:
    """Configuration for surrogate visual-class assignment."""

    kmeans_clusters: int = 5000
    pca_dim: int = 32
    ratio_threshold: float = 0.9
    assignment_chunk_size: int = 4096
    patch_size: int = 32


ASSIGNMENT_COLUMNS = [
    "assignment_row",
    "descriptor_row",
    "page_descriptor_row",
    "local_keypoint_index",
    "split",
    "writer_id",
    "image_id",
    "filename",
    "image_path",
    "patch_size",
    "kp_x",
    "kp_y",
    "kp_size",
    "kp_angle",
    "kp_response",
    "foreground_ratio",
    "surrogate_label",
    "ratio",
    "nearest_distance",
    "second_distance",
]


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def atomic_savez_compressed(path: Path, **arrays: Any) -> None:
    """Write an npz atomically using a temporary file followed by os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, path)


def load_train_page_stats(stage01_root: Path) -> pd.DataFrame:
    """Load Stage 01 TRAIN page stats."""
    path = stage01_root / "train" / "train_page_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing TRAIN page stats: {path}")

    df = pd.read_csv(path)

    required = {
        "image_id",
        "filename",
        "image_path",
        "writer_id",
        "split",
        "total_keypoints_kept",
        "page_npz_path",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["filename"] = df["filename"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    df["writer_id"] = pd.to_numeric(df["writer_id"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str)
    df["total_keypoints_kept"] = pd.to_numeric(
        df["total_keypoints_kept"], errors="raise"
    ).astype(int)

    return df


def build_page_descriptor_ranges(page_stats: pd.DataFrame) -> pd.DataFrame:
    """Add descriptor start/end rows per page based on total_keypoints_kept."""
    out = page_stats.copy()
    counts = out["total_keypoints_kept"].astype(int).to_numpy()

    starts = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        starts[1:] = np.cumsum(counts[:-1], dtype=np.int64)

    ends = starts + counts

    out["descriptor_start"] = starts
    out["descriptor_end"] = ends
    return out


def apply_smoke_subset(df: pd.DataFrame, max_pages: int) -> pd.DataFrame:
    """Return a deterministic small subset for smoke testing."""
    if max_pages <= 0:
        return df

    out = df.head(max_pages).copy()
    print(f"[SMOKE] train: using {len(out)} / {len(df)} pages")
    return out


def load_train_pca32(stage02_root: Path) -> np.ndarray:
    """Load TRAIN PCA-32 descriptors as memmap."""
    path = stage02_root / "train" / "train_rootsift_pca32.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing TRAIN PCA-32 descriptors: {path}")

    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 32:
        raise ValueError(f"Expected TRAIN PCA-32 shape [N, 32], got {arr.shape}")

    return arr


def load_kmeans_centers(stage02_root: Path, cfg: Stage03Config) -> np.ndarray:
    """Load KMeans centers."""
    path = (
        stage02_root
        / "kmeans"
        / f"minibatch_kmeans_k{cfg.kmeans_clusters}_centers.npy"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing KMeans centers: {path}")

    centers = np.load(path)
    expected = (cfg.kmeans_clusters, cfg.pca_dim)
    if centers.shape != expected:
        raise ValueError(f"Expected centers shape {expected}, got {centers.shape}")

    if not np.isfinite(centers).all():
        raise ValueError("KMeans centers contain non-finite values")

    return centers.astype(np.float32, copy=False)


def page_assignment_path(pages_dir: Path, image_id: str) -> Path:
    """Return per-page assignment checkpoint path."""
    return pages_dir / f"{image_id}.npz"


def page_assignment_is_valid(path: Path, cfg: Stage03Config) -> bool:
    """Check whether an existing per-page assignment checkpoint is valid."""
    if not path.exists():
        return False

    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "descriptor_row",
                "page_descriptor_row",
                "local_keypoint_index",
                "surrogate_label",
                "ratio",
                "nearest_distance",
                "second_distance",
                "kp_x",
                "kp_y",
                "kp_size",
                "kp_angle",
                "kp_response",
                "foreground_ratio",
                "stats_json",
            }
            if not required.issubset(set(data.files)):
                return False

            labels = data["surrogate_label"]
            ratio = data["ratio"]
            d1 = data["nearest_distance"]
            d2 = data["second_distance"]

            n = len(labels)
            same_length = (
                len(data["descriptor_row"]) == n
                and len(data["page_descriptor_row"]) == n
                and len(data["local_keypoint_index"]) == n
                and len(ratio) == n
                and len(d1) == n
                and len(d2) == n
            )
            if not same_length:
                return False

            if n > 0:
                if labels.min() < 0 or labels.max() >= cfg.kmeans_clusters:
                    return False
                if not np.isfinite(ratio).all():
                    return False
                if not np.isfinite(d1).all() or not np.isfinite(d2).all():
                    return False
                if np.any(ratio > cfg.ratio_threshold + 1e-6):
                    return False

        return True
    except Exception:
        return False


def load_stage01_page_npz(stage01_page_npz: Path, expected_count: int) -> Dict[str, Any]:
    """Load needed per-page metadata arrays from a Stage 01 checkpoint."""
    if not stage01_page_npz.exists():
        raise FileNotFoundError(f"Missing Stage 01 page checkpoint: {stage01_page_npz}")

    with np.load(stage01_page_npz, allow_pickle=False) as data:
        raw = data["raw_descriptors"]
        if raw.shape[0] != expected_count:
            raise ValueError(
                f"Stage 01 page descriptor count mismatch for {stage01_page_npz}: "
                f"{raw.shape[0]} != {expected_count}"
            )

        out = {
            "page_descriptor_row": data["page_descriptor_row"].astype(np.int64),
            "local_keypoint_index": data["local_keypoint_index"].astype(np.int64),
            "kp_x": data["kp_x"].astype(np.float32),
            "kp_y": data["kp_y"].astype(np.float32),
            "kp_size": data["kp_size"].astype(np.float32),
            "kp_angle": data["kp_angle"].astype(np.float32),
            "kp_response": data["kp_response"].astype(np.float32),
            "foreground_ratio": data["foreground_ratio"].astype(np.float32),
        }

    for key, arr in out.items():
        if len(arr) != expected_count:
            raise ValueError(
                f"Stage 01 metadata array length mismatch for {stage01_page_npz}, "
                f"{key}: {len(arr)} != {expected_count}"
            )

    return out


def nearest_two_centers_chunked(
    x: np.ndarray,
    centers: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find nearest and second-nearest centers using chunked squared L2 distances."""
    if x.ndim != 2:
        raise ValueError(f"Expected x [N, D], got {x.shape}")
    if centers.ndim != 2:
        raise ValueError(f"Expected centers [K, D], got {centers.shape}")
    if x.shape[1] != centers.shape[1]:
        raise ValueError(f"Descriptor dim {x.shape[1]} != center dim {centers.shape[1]}")
    if centers.shape[0] < 2:
        raise ValueError("Need at least two centers for ratio assignment")

    x = np.asarray(x, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)

    n = x.shape[0]
    distances = np.empty((n, 2), dtype=np.float32)
    indices = np.empty((n, 2), dtype=np.int64)

    centers_t = centers.T.copy()
    centers_norm = np.sum(centers * centers, axis=1).reshape(1, -1)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xb = x[start:end]

        xb_norm = np.sum(xb * xb, axis=1).reshape(-1, 1)
        dist2 = xb_norm + centers_norm - 2.0 * (xb @ centers_t)
        np.maximum(dist2, 0.0, out=dist2)

        idx_unsorted = np.argpartition(dist2, kth=1, axis=1)[:, :2]
        val_unsorted = np.take_along_axis(dist2, idx_unsorted, axis=1)

        order = np.argsort(val_unsorted, axis=1)
        idx_sorted = np.take_along_axis(idx_unsorted, order, axis=1)
        val_sorted = np.take_along_axis(val_unsorted, order, axis=1)

        distances[start:end] = np.sqrt(val_sorted).astype(np.float32)
        indices[start:end] = idx_sorted.astype(np.int64)

    return distances, indices


def extract_one_page_assignments(
    row: Any,
    train_pca32: np.ndarray,
    centers: np.ndarray,
    cfg: Stage03Config,
    pages_dir: Path,
    overwrite_pages: bool,
) -> Tuple[str, Dict[str, Any]]:
    """Build and save surrogate assignments for one page."""
    image_id = str(row.image_id)
    filename = str(row.filename)
    image_path = str(row.image_path)
    writer_id = int(row.writer_id)
    split = str(row.split)

    start = int(row.descriptor_start)
    end = int(row.descriptor_end)
    count = int(row.total_keypoints_kept)

    if end - start != count:
        raise ValueError(f"Invalid descriptor range for {image_id}: {start}:{end}, count={count}")

    out_path = page_assignment_path(pages_dir, image_id)
    if page_assignment_is_valid(out_path, cfg) and not overwrite_pages:
        with np.load(out_path, allow_pickle=False) as data:
            stats = json.loads(str(data["stats_json"].item()))
        return "skipped", stats

    page_npz_path = Path(str(row.page_npz_path))
    if not page_npz_path.is_absolute():
        page_npz_path = PROJECT_ROOT / page_npz_path

    meta = load_stage01_page_npz(page_npz_path, expected_count=count)

    if count > 0:
        x = np.asarray(train_pca32[start:end], dtype=np.float32)

        if x.shape != (count, cfg.pca_dim):
            raise ValueError(f"Unexpected PCA page shape for {image_id}: {x.shape}")

        dists, idxs = nearest_two_centers_chunked(
            x=x,
            centers=centers,
            chunk_size=cfg.assignment_chunk_size,
        )

        d1 = dists[:, 0]
        d2 = dists[:, 1]
        labels = idxs[:, 0]

        ratio = d1 / np.maximum(d2, np.float32(1e-12))
        accepted_mask = ratio <= cfg.ratio_threshold

        page_rows = meta["page_descriptor_row"][accepted_mask]
        local_indices = meta["local_keypoint_index"][accepted_mask]
        global_rows = (start + np.arange(count, dtype=np.int64))[accepted_mask]
        accepted_labels = labels[accepted_mask].astype(np.int64)
        accepted_ratio = ratio[accepted_mask].astype(np.float32)
        accepted_d1 = d1[accepted_mask].astype(np.float32)
        accepted_d2 = d2[accepted_mask].astype(np.float32)

        kp_x = meta["kp_x"][accepted_mask]
        kp_y = meta["kp_y"][accepted_mask]
        kp_size = meta["kp_size"][accepted_mask]
        kp_angle = meta["kp_angle"][accepted_mask]
        kp_response = meta["kp_response"][accepted_mask]
        foreground_ratio = meta["foreground_ratio"][accepted_mask]
    else:
        page_rows = np.empty((0,), dtype=np.int64)
        local_indices = np.empty((0,), dtype=np.int64)
        global_rows = np.empty((0,), dtype=np.int64)
        accepted_labels = np.empty((0,), dtype=np.int64)
        accepted_ratio = np.empty((0,), dtype=np.float32)
        accepted_d1 = np.empty((0,), dtype=np.float32)
        accepted_d2 = np.empty((0,), dtype=np.float32)
        kp_x = np.empty((0,), dtype=np.float32)
        kp_y = np.empty((0,), dtype=np.float32)
        kp_size = np.empty((0,), dtype=np.float32)
        kp_angle = np.empty((0,), dtype=np.float32)
        kp_response = np.empty((0,), dtype=np.float32)
        foreground_ratio = np.empty((0,), dtype=np.float32)

    accepted_count = int(len(accepted_labels))
    rejected_ratio = int(count - accepted_count)

    stats = {
        "image_id": image_id,
        "filename": filename,
        "image_path": image_path,
        "writer_id": writer_id,
        "split": split,
        "descriptor_start": int(start),
        "descriptor_end": int(end),
        "total_descriptors": int(count),
        "accepted_descriptors": int(accepted_count),
        "rejected_by_ratio": int(rejected_ratio),
        "acceptance_rate": float(accepted_count / count) if count > 0 else 0.0,
        "ratio_threshold": float(cfg.ratio_threshold),
        "kmeans_clusters": int(cfg.kmeans_clusters),
        "pca_dim": int(cfg.pca_dim),
        "assignment_npz_path": str(out_path),
    }

    atomic_savez_compressed(
        out_path,
        descriptor_row=global_rows,
        page_descriptor_row=page_rows,
        local_keypoint_index=local_indices,
        surrogate_label=accepted_labels,
        ratio=accepted_ratio,
        nearest_distance=accepted_d1,
        second_distance=accepted_d2,
        kp_x=kp_x.astype(np.float32),
        kp_y=kp_y.astype(np.float32),
        kp_size=kp_size.astype(np.float32),
        kp_angle=kp_angle.astype(np.float32),
        kp_response=kp_response.astype(np.float32),
        foreground_ratio=foreground_ratio.astype(np.float32),
        image_id=np.asarray(image_id),
        filename=np.asarray(filename),
        image_path=np.asarray(image_path),
        writer_id=np.asarray(writer_id, dtype=np.int64),
        split=np.asarray(split),
        patch_size=np.asarray(cfg.patch_size, dtype=np.int64),
        stats_json=np.asarray(json.dumps(stats, ensure_ascii=False)),
    )

    return "processed", stats


def write_assignment_rows_from_page(
    writer: csv.DictWriter,
    data: Any,
    assignment_start_row: int,
) -> int:
    """Write accepted assignment metadata rows from a page checkpoint."""
    n = int(len(data["surrogate_label"]))

    image_id = str(data["image_id"].item())
    filename = str(data["filename"].item())
    image_path = str(data["image_path"].item())
    writer_id = int(data["writer_id"].item())
    split = str(data["split"].item())
    patch_size = int(data["patch_size"].item())

    for i in range(n):
        writer.writerow(
            {
                "assignment_row": int(assignment_start_row + i),
                "descriptor_row": int(data["descriptor_row"][i]),
                "page_descriptor_row": int(data["page_descriptor_row"][i]),
                "local_keypoint_index": int(data["local_keypoint_index"][i]),
                "split": split,
                "writer_id": writer_id,
                "image_id": image_id,
                "filename": filename,
                "image_path": image_path,
                "patch_size": patch_size,
                "kp_x": float(data["kp_x"][i]),
                "kp_y": float(data["kp_y"][i]),
                "kp_size": float(data["kp_size"][i]),
                "kp_angle": float(data["kp_angle"][i]),
                "kp_response": float(data["kp_response"][i]),
                "foreground_ratio": float(data["foreground_ratio"][i]),
                "surrogate_label": int(data["surrogate_label"][i]),
                "ratio": float(data["ratio"][i]),
                "nearest_distance": float(data["nearest_distance"][i]),
                "second_distance": float(data["second_distance"][i]),
            }
        )

    return n


def aggregate_train_assignments(
    page_ranges: pd.DataFrame,
    output_root: Path,
    cfg: Stage03Config,
    overwrite_aggregate: bool,
) -> Dict[str, Any]:
    """Aggregate per-page assignment checkpoints into global files."""
    train_root = output_root / "train"
    pages_dir = train_root / "pages"

    csv_path = train_root / "train_surrogate_assignments.csv"
    page_stats_path = train_root / "train_surrogate_page_stats.csv"
    accepted_rows_path = train_root / "train_accepted_descriptor_rows.npy"
    labels_path = train_root / "train_surrogate_labels.npy"
    ratios_path = train_root / "train_assignment_ratios.npy"
    cluster_hist_path = train_root / "train_cluster_histogram_k5000.npy"
    summary_path = train_root / "train_surrogate_assignment_summary.json"

    outputs_exist = (
        csv_path.exists()
        and page_stats_path.exists()
        and accepted_rows_path.exists()
        and labels_path.exists()
        and ratios_path.exists()
        and cluster_hist_path.exists()
    )
    if outputs_exist and not overwrite_aggregate:
        print("[AGG] train aggregate outputs already exist; skipping aggregation.")
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    page_paths: List[Path] = []
    stats_rows: List[Dict[str, Any]] = []
    total_accepted = 0
    total_descriptors = 0

    for row in page_ranges.itertuples(index=False):
        path = page_assignment_path(pages_dir, str(row.image_id))
        if not page_assignment_is_valid(path, cfg):
            raise FileNotFoundError(f"Missing or invalid page assignment checkpoint: {path}")

        with np.load(path, allow_pickle=False) as data:
            n = int(len(data["surrogate_label"]))
            stats = json.loads(str(data["stats_json"].item()))

        page_paths.append(path)
        stats_rows.append(stats)
        total_accepted += n
        total_descriptors += int(stats["total_descriptors"])

    print(
        f"[AGG] train: pages={len(page_paths)}, "
        f"total_descriptors={total_descriptors}, accepted={total_accepted}"
    )

    tmp_csv = csv_path.with_name(csv_path.name + ".tmp")
    tmp_page_stats = page_stats_path.with_name(page_stats_path.name + ".tmp")
    tmp_rows = accepted_rows_path.with_name(accepted_rows_path.name + ".tmp")
    tmp_labels = labels_path.with_name(labels_path.name + ".tmp")
    tmp_ratios = ratios_path.with_name(ratios_path.name + ".tmp")
    tmp_hist = cluster_hist_path.with_name(cluster_hist_path.name + ".tmp")

    for tmp in [tmp_csv, tmp_page_stats, tmp_rows, tmp_labels, tmp_ratios, tmp_hist]:
        if tmp.exists():
            tmp.unlink()

    rows_mmap = np.lib.format.open_memmap(
        tmp_rows,
        mode="w+",
        dtype=np.int64,
        shape=(total_accepted,),
    )
    labels_mmap = np.lib.format.open_memmap(
        tmp_labels,
        mode="w+",
        dtype=np.int64,
        shape=(total_accepted,),
    )
    ratios_mmap = np.lib.format.open_memmap(
        tmp_ratios,
        mode="w+",
        dtype=np.float32,
        shape=(total_accepted,),
    )

    cluster_hist = np.zeros(cfg.kmeans_clusters, dtype=np.int64)

    assignment_row = 0
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ASSIGNMENT_COLUMNS)
        writer.writeheader()

        for path in tqdm(page_paths, desc="Aggregate train assignments"):
            with np.load(path, allow_pickle=False) as data:
                labels = data["surrogate_label"].astype(np.int64)
                n = int(len(labels))

                rows_mmap[assignment_row : assignment_row + n] = data["descriptor_row"]
                labels_mmap[assignment_row : assignment_row + n] = labels
                ratios_mmap[assignment_row : assignment_row + n] = data["ratio"]

                if n > 0:
                    cluster_hist += np.bincount(labels, minlength=cfg.kmeans_clusters)

                written = write_assignment_rows_from_page(
                    writer=writer,
                    data=data,
                    assignment_start_row=assignment_row,
                )
                if written != n:
                    raise ValueError(f"CSV row count mismatch for {path}: {written} != {n}")

                assignment_row += n

    rows_mmap.flush()
    labels_mmap.flush()
    ratios_mmap.flush()
    del rows_mmap
    del labels_mmap
    del ratios_mmap

    pd.DataFrame(stats_rows).to_csv(tmp_page_stats, index=False)

    with tmp_hist.open("wb") as f:
        np.save(f, cluster_hist)

    os.replace(tmp_csv, csv_path)
    os.replace(tmp_page_stats, page_stats_path)
    os.replace(tmp_rows, accepted_rows_path)
    os.replace(tmp_labels, labels_path)
    os.replace(tmp_ratios, ratios_path)
    os.replace(tmp_hist, cluster_hist_path)

    non_empty_clusters = int((cluster_hist > 0).sum())
    empty_clusters = int((cluster_hist == 0).sum())

    summary = {
        "stage": "03R_build_surrogate_patch_dataset_rsift_random500k",
        "split": "train",
        "num_pages": int(len(page_ranges)),
        "total_descriptors": int(total_descriptors),
        "accepted_descriptors": int(total_accepted),
        "rejected_by_ratio": int(total_descriptors - total_accepted),
        "acceptance_rate": float(total_accepted / total_descriptors)
        if total_descriptors > 0
        else 0.0,
        "ratio_threshold": float(cfg.ratio_threshold),
        "kmeans_clusters": int(cfg.kmeans_clusters),
        "non_empty_surrogate_labels": non_empty_clusters,
        "empty_surrogate_labels": empty_clusters,
        "min_non_empty_label_count": int(cluster_hist[cluster_hist > 0].min())
        if non_empty_clusters > 0
        else 0,
        "max_label_count": int(cluster_hist.max()) if len(cluster_hist) else 0,
        "mean_label_count": float(cluster_hist.mean()) if len(cluster_hist) else 0.0,
        "outputs": {
            "assignments_csv": str(csv_path),
            "page_stats": str(page_stats_path),
            "accepted_descriptor_rows": str(accepted_rows_path),
            "surrogate_labels": str(labels_path),
            "assignment_ratios": str(ratios_path),
            "cluster_histogram": str(cluster_hist_path),
            "pages_dir": str(pages_dir),
        },
        "config": asdict(cfg),
    }

    atomic_write_json(summary_path, summary)

    print("[OK] train assignment aggregation complete")
    print(f"[OUT] {csv_path}")
    print(f"[OUT] {accepted_rows_path}")
    print(f"[OUT] {labels_path}")
    print(f"[OUT] {ratios_path}")
    print(f"[OUT] {cluster_hist_path}")

    return summary


def run_train_assignments(
    stage01_root: Path,
    stage02_root: Path,
    output_root: Path,
    cfg: Stage03Config,
    smoke_test: bool,
    max_pages: int,
    overwrite_pages: bool,
    overwrite_aggregate: bool,
    aggregate: bool,
    aggregate_only: bool,
) -> Dict[str, Any]:
    """Run resumable TRAIN surrogate assignment."""
    output_root.mkdir(parents=True, exist_ok=True)
    train_root = output_root / "train"
    pages_dir = train_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    page_stats = load_train_page_stats(stage01_root)
    page_ranges = build_page_descriptor_ranges(page_stats)

    train_pca32 = load_train_pca32(stage02_root)

    expected_rows = int(page_ranges["total_keypoints_kept"].sum())
    if expected_rows != train_pca32.shape[0]:
        raise ValueError(
            f"TRAIN page stats sum {expected_rows} != PCA-32 rows {train_pca32.shape[0]}"
        )

    if smoke_test:
        page_ranges = apply_smoke_subset(page_ranges, max_pages=max_pages)

    centers = load_kmeans_centers(stage02_root, cfg)

    all_summary: Dict[str, Any] = {
        "stage": "03R_build_surrogate_patch_dataset_rsift_random500k",
        "split": "train",
        "stage01_root": str(stage01_root),
        "stage02_root": str(stage02_root),
        "output_root": str(output_root),
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "methodological_status": "surrogate_visual_labels_from_rsift_pca32_kmeans_ratio_test",
        "surrogate_assignment_source": "TRAIN_ONLY",
        "test_used_for_assignment": False,
        "ratio_test": {
            "accepted_if": "d1 / d2 <= ratio_threshold",
            "ratio_threshold": float(cfg.ratio_threshold),
            "kmeans_clusters": int(cfg.kmeans_clusters),
            "pca_dim": int(cfg.pca_dim),
        },
        "config": asdict(cfg),
        "smoke_test": bool(smoke_test),
        "max_pages": int(max_pages) if smoke_test else None,
        "aggregate": bool(aggregate),
        "aggregate_only": bool(aggregate_only),
    }

    progress_path = train_root / "train_assignment_progress.json"

    if not aggregate_only:
        processed = 0
        skipped = 0
        failed = 0

        for page_idx, row in enumerate(
            tqdm(
                page_ranges.itertuples(index=False),
                total=len(page_ranges),
                desc="Assign train",
            )
        ):
            try:
                status, stats = extract_one_page_assignments(
                    row=row,
                    train_pca32=train_pca32,
                    centers=centers,
                    cfg=cfg,
                    pages_dir=pages_dir,
                    overwrite_pages=overwrite_pages,
                )

                if status == "processed":
                    processed += 1
                elif status == "skipped":
                    skipped += 1

            except Exception as exc:
                failed += 1
                atomic_write_json(
                    progress_path,
                    {
                        "status": "failed",
                        "page_idx": int(page_idx),
                        "processed_pages_this_run": int(processed),
                        "skipped_pages_this_run": int(skipped),
                        "failed_pages_this_run": int(failed),
                        "error": repr(exc),
                    },
                )
                raise

            if (page_idx + 1) % 50 == 0 or (page_idx + 1) == len(page_ranges):
                atomic_write_json(
                    progress_path,
                    {
                        "status": "running",
                        "pages_seen": int(page_idx + 1),
                        "total_pages": int(len(page_ranges)),
                        "processed_pages_this_run": int(processed),
                        "skipped_pages_this_run": int(skipped),
                        "failed_pages_this_run": int(failed),
                        "last_image_id": str(row.image_id),
                    },
                )
                print(
                    f"[train] pages_seen={page_idx + 1}/{len(page_ranges)} | "
                    f"processed={processed} | skipped={skipped} | failed={failed}"
                )

        atomic_write_json(
            progress_path,
            {
                "status": "page_assignment_complete",
                "total_pages": int(len(page_ranges)),
                "processed_pages_this_run": int(processed),
                "skipped_pages_this_run": int(skipped),
                "failed_pages_this_run": int(failed),
                "pages_dir": str(pages_dir),
            },
        )

    if aggregate:
        aggregate_summary = aggregate_train_assignments(
            page_ranges=page_ranges,
            output_root=output_root,
            cfg=cfg,
            overwrite_aggregate=overwrite_aggregate,
        )
        all_summary["aggregate_summary"] = aggregate_summary

        atomic_write_json(
            progress_path,
            {
                "status": "complete_with_aggregation",
                "total_pages": int(len(page_ranges)),
                "pages_dir": str(pages_dir),
            },
        )

    atomic_write_json(output_root / "stage03_last_run_summary.json", all_summary)
    print("[OK] Stage 03 completed")

    return all_summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 03R: strict-branch surrogate visual-class assignment using R-SIFT PCA32 descriptors and KMeans ratio test."
    )

    parser.add_argument("--stage01-root", type=str, default=str(DEFAULT_STAGE01_ROOT))
    parser.add_argument("--stage02-root", type=str, default=str(DEFAULT_STAGE02_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))

    parser.add_argument("--ratio-threshold", type=float, default=0.9)
    parser.add_argument("--kmeans-clusters", type=int, default=5000)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--assignment-chunk-size", type=int, default=4096)
    parser.add_argument("--patch-size", type=int, default=32)

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-pages", type=int, default=10)

    parser.add_argument("--overwrite-pages", action="store_true")
    parser.add_argument("--overwrite-aggregate", action="store_true")

    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Run Stage 03."""
    args = parse_args()

    cfg = Stage03Config(
        kmeans_clusters=int(args.kmeans_clusters),
        pca_dim=int(args.pca_dim),
        ratio_threshold=float(args.ratio_threshold),
        assignment_chunk_size=int(args.assignment_chunk_size),
        patch_size=int(args.patch_size),
    )

    print("[CONFIG]")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    print(f"[STAGE01_ROOT] {args.stage01_root}")
    print(f"[STAGE02_ROOT] {args.stage02_root}")
    print(f"[OUTPUT_ROOT] {args.output_root}")

    run_train_assignments(
        stage01_root=Path(args.stage01_root),
        stage02_root=Path(args.stage02_root),
        output_root=Path(args.output_root),
        cfg=cfg,
        smoke_test=bool(args.smoke_test),
        max_pages=int(args.max_pages),
        overwrite_pages=bool(args.overwrite_pages),
        overwrite_aggregate=bool(args.overwrite_aggregate),
        aggregate=not bool(args.no_aggregate),
        aggregate_only=bool(args.aggregate_only),
    )


if __name__ == "__main__":
    main()