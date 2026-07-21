from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(".")

INPUT_MANIFESTS_DIR = PROJECT_ROOT / "outputs" / "faithful_christlein2017" / "manifests"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "faithful_christlein2017"
    / "strict_christlein2017_rsift_random500k"
    / "rsift_rootsift_patches"
)

TRAIN_MANIFEST = INPUT_MANIFESTS_DIR / "train_manifest.csv"
TEST_MANIFEST = INPUT_MANIFESTS_DIR / "test_manifest.csv"


@dataclass
class ExtractionConfig:
    """Configuration for faithful-compatible SIFT / RootSIFT / patch extraction."""

    patch_size: int = 32

    # Faithful default: 0 means no artificial per-page cap.
    # The previous Christlein-inspired pipeline used 1200.
    max_keypoints_per_page: int = 0

    sift_nfeatures: int = 0
    sift_noctave_layers: int = 3
    sift_contrast_threshold: float = 0.005
    sift_edge_threshold: float = 10.0
    sift_sigma: float = 1.6

    # Python/OpenCV approximation of restricted SIFT on binarized text regions.
    foreground_only: bool = True
    foreground_threshold: int = 200
    min_foreground_ratio_in_patch: float = 0.05

    # R-SIFT emulation settings.
    # The image is internally converted to canonical dark-ink-on-bright-background.
    # Keypoints are then filtered so that retained patches contain ink and the
    # keypoint neighborhood is close to ink strokes.
    rsift_center_window: int = 7
    rsift_min_center_ink_ratio: float = 0.02
    rsift_dedup_location_round_decimals: int = 0

    # Faithful default: False. Fixed-angle was used in the previous inspired pipeline,
    # but it is not an explicit requirement of Christlein et al. (2017).
    fixed_angle: bool = False

    dedup_patches_per_page: bool = True

    # For no-cap faithful extraction, patch images should normally not be saved.
    save_patch_images: bool = False
    patch_image_ext: str = ".png"


@dataclass
class PageExtractionStats:
    """Per-page extraction statistics."""

    image_id: str
    filename: str
    image_path: str
    writer_id: int
    split: str
    total_keypoints_raw: int
    total_keypoints_after_cap: int
    total_keypoints_kept: int
    rejected_border: int
    rejected_low_foreground: int
    rejected_dedup: int
    page_npz_path: str


METADATA_COLUMNS = [
    "descriptor_row",
    "page_descriptor_row",
    "split",
    "writer_id",
    "image_id",
    "filename",
    "image_path",
    "patch_name",
    "patch_path",
    "local_keypoint_index",
    "kp_x",
    "kp_y",
    "kp_size",
    "kp_angle",
    "kp_response",
    "foreground_ratio",
]


def load_grayscale_image(image_path: str) -> np.ndarray:
    """Load an image as grayscale uint8."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def canonical_dark_ink_image(img: np.ndarray) -> np.ndarray:
    """Return image in canonical dark-ink-on-bright-background form.

    The dataset image may be stored either as dark ink on bright background
    or as bright foreground on dark background. For Stage 01R we work
    internally with dark ink and bright background.

    The decision is based on the page border, which is normally background.
    If the border is dark, the page is inverted.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected grayscale image, got shape {img.shape}")

    h, w = img.shape
    border = np.concatenate([
        img[0, :],
        img[h - 1, :],
        img[:, 0],
        img[:, w - 1],
    ])
    border_median = float(np.median(border))

    if border_median < 128.0:
        return cv2.bitwise_not(img)

    return img


def make_foreground_mask(img: np.ndarray, foreground_threshold: int) -> np.ndarray:
    """Build an ink mask for canonical dark-ink-on-bright-background images.

    The function name is kept for compatibility with the previous Stage 01
    output logic. In Stage 01R this mask is not passed to SIFT detection.
    """
    return ((img <= foreground_threshold).astype(np.uint8) * 255)


def create_sift_detector(cfg: ExtractionConfig):
    """Create an OpenCV SIFT detector with explicit parameters."""
    return cv2.SIFT_create(
        nfeatures=cfg.sift_nfeatures,
        nOctaveLayers=cfg.sift_noctave_layers,
        contrastThreshold=cfg.sift_contrast_threshold,
        edgeThreshold=cfg.sift_edge_threshold,
        sigma=cfg.sift_sigma,
    )


def detect_keypoints_and_descriptors(
    img: np.ndarray,
    foreground_mask: Optional[np.ndarray],
    sift,
    fixed_angle: bool,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """Detect SIFT keypoints and descriptors."""
    keypoints, descriptors = sift.detectAndCompute(img, foreground_mask)
    if descriptors is None or len(keypoints) == 0:
        return [], None

    if not fixed_angle:
        return list(keypoints), descriptors.astype(np.float32)

    fixed_kps: List[cv2.KeyPoint] = []
    for kp in keypoints:
        fixed_kps.append(
            cv2.KeyPoint(
                x=kp.pt[0],
                y=kp.pt[1],
                size=kp.size,
                angle=0.0,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id,
            )
        )

    fixed_kps, fixed_desc = sift.compute(img, fixed_kps)
    if fixed_desc is None or len(fixed_kps) == 0:
        return [], None

    return list(fixed_kps), fixed_desc.astype(np.float32)


def sort_keypoints_by_response(
    keypoints: List[cv2.KeyPoint],
    descriptors: np.ndarray,
    max_keypoints_per_page: int,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Sort keypoints by descending response and optionally cap per page."""
    if descriptors is None or len(keypoints) == 0:
        return [], np.empty((0, 128), dtype=np.float32)

    pairs = list(zip(keypoints, descriptors))
    pairs.sort(key=lambda x: float(x[0].response), reverse=True)

    if max_keypoints_per_page > 0:
        pairs = pairs[:max_keypoints_per_page]

    out_kps = [p[0] for p in pairs]
    out_desc = np.stack([p[1] for p in pairs], axis=0).astype(np.float32)
    return out_kps, out_desc


def dedup_keypoints_by_location(
    keypoints: List[cv2.KeyPoint],
    descriptors: np.ndarray,
    round_decimals: int,
) -> Tuple[List[cv2.KeyPoint], np.ndarray, int]:
    """Keep one keypoint per rounded image location.

    Keypoints are expected to be sorted by descending response. Therefore,
    if duplicates occur at the same rounded location, the strongest response
    is retained.
    """
    if descriptors is None or len(keypoints) == 0:
        return [], np.empty((0, 128), dtype=np.float32), 0

    kept_kps: List[cv2.KeyPoint] = []
    kept_desc: List[np.ndarray] = []
    seen = set()
    rejected = 0

    for kp, desc in zip(keypoints, descriptors):
        loc = (
            round(float(kp.pt[0]), int(round_decimals)),
            round(float(kp.pt[1]), int(round_decimals)),
        )
        if loc in seen:
            rejected += 1
            continue
        seen.add(loc)
        kept_kps.append(kp)
        kept_desc.append(desc)

    if kept_desc:
        out_desc = np.stack(kept_desc, axis=0).astype(np.float32)
    else:
        out_desc = np.empty((0, 128), dtype=np.float32)

    return kept_kps, out_desc, rejected


def crop_patch_centered(
    img: np.ndarray,
    x: float,
    y: float,
    patch_size: int,
) -> Optional[np.ndarray]:
    """Extract a square patch centered at (x, y)."""
    half = patch_size // 2
    cx = int(round(x))
    cy = int(round(y))

    x0 = cx - half
    y0 = cy - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        return None

    patch = img[y0:y1, x0:x1]
    if patch.shape != (patch_size, patch_size):
        return None

    return patch


def patch_foreground_ratio(patch: np.ndarray, foreground_threshold: int) -> float:
    """Compute the fraction of dark ink pixels in a canonical R-SIFT patch.

    The output field name foreground_ratio is preserved for schema
    compatibility with the previous Stage 01 implementation.
    """
    ink = (patch <= foreground_threshold).astype(np.float32)
    return float(ink.mean())


def patch_center_ink_ratio(
    patch: np.ndarray,
    foreground_threshold: int,
    center_window: int,
) -> float:
    """Compute the ink ratio in a small window around the patch center."""
    if center_window <= 0:
        raise ValueError("center_window must be positive")

    h, w = patch.shape
    cx = w // 2
    cy = h // 2
    half = center_window // 2

    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)

    center = patch[y0:y1, x0:x1]
    ink = (center <= foreground_threshold).astype(np.float32)
    return float(ink.mean())


def patch_sha1(patch: np.ndarray) -> str:
    """Compute a stable hash for per-page patch deduplication."""
    return hashlib.sha1(patch.tobytes()).hexdigest()


def rootsift_descriptors(descriptors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert SIFT descriptors to RootSIFT descriptors."""
    if descriptors.ndim != 2 or descriptors.shape[1] != 128:
        raise ValueError(f"Expected descriptors with shape [N, 128], got {descriptors.shape}")

    x = descriptors.astype(np.float32, copy=True)
    x = np.maximum(x, 0.0)

    l1 = np.sum(x, axis=1, keepdims=True)
    x = x / np.maximum(l1, eps)

    x = np.sqrt(x)

    l2 = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(l2, eps)

    return x.astype(np.float32)


def load_manifest(path: Path) -> pd.DataFrame:
    """Load a validated manifest produced by 00_prepare_manifests.py."""
    if not path.exists():
        raise FileNotFoundError(f"Validated manifest not found: {path}")

    df = pd.read_csv(path)

    required = {"image_id", "filename", "writer_id", "split", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")

    df = df[["image_id", "filename", "writer_id", "split", "image_path"]].copy()
    df["image_id"] = df["image_id"].astype(str)
    df["filename"] = df["filename"].astype(str)
    df["writer_id"] = pd.to_numeric(df["writer_id"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str).str.lower()
    df["image_path"] = df["image_path"].astype(str)

    return df


def apply_smoke_subset(
    df: pd.DataFrame,
    split: str,
    max_pages_per_split: int,
) -> pd.DataFrame:
    """Return a small deterministic subset for smoke testing."""
    if max_pages_per_split <= 0:
        return df

    out = df.head(max_pages_per_split).copy()
    print(f"[SMOKE] {split}: using {len(out)} / {len(df)} pages")
    return out


def page_npz_is_valid(path: Path) -> bool:
    """Check whether an existing per-page checkpoint is readable and structurally valid."""
    if not path.exists():
        return False

    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "raw_descriptors",
                "rootsift_descriptors",
                "page_descriptor_row",
                "local_keypoint_index",
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

            raw = data["raw_descriptors"]
            root = data["rootsift_descriptors"]

            if raw.ndim != 2 or root.ndim != 2:
                return False
            if raw.shape != root.shape:
                return False
            if raw.shape[1] != 128:
                return False
            if not np.isfinite(raw).all():
                return False
            if not np.isfinite(root).all():
                return False

        return True
    except Exception:
        return False


def atomic_savez_compressed(path: Path, **arrays: Any) -> None:
    """Write an npz atomically using a temporary file followed by os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")

    if tmp_path.exists():
        tmp_path.unlink()

    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **arrays)

    os.replace(tmp_path, path)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    os.replace(tmp_path, path)


def extract_one_page(
    row: Any,
    split: str,
    cfg: ExtractionConfig,
    sift,
    pages_dir: Path,
    patches_dir: Path,
    overwrite_pages: bool,
) -> Tuple[str, bool, PageExtractionStats]:
    """Extract one page and save a resumable per-page checkpoint."""
    image_id = str(row.image_id)
    filename = str(row.filename)
    image_path = str(row.image_path)
    writer_id = int(row.writer_id)
    split_name = str(row.split)

    page_npz_path = pages_dir / f"{image_id}.npz"

    if page_npz_is_valid(page_npz_path) and not overwrite_pages:
        with np.load(page_npz_path, allow_pickle=False) as data:
            stats = json.loads(str(data["stats_json"].item()))
        return "skipped", False, PageExtractionStats(**stats)

    img = load_grayscale_image(image_path)
    img = canonical_dark_ink_image(img)

    # Stage 01R does not use the previous simple foreground mask for SIFT
    # detection. Instead, it detects SIFT keypoints on the canonical
    # dark-ink-on-bright-background image and then filters keypoints/patches
    # according to R-SIFT-like ink-content criteria.
    fg_mask = None

    keypoints, descriptors = detect_keypoints_and_descriptors(
        img=img,
        foreground_mask=fg_mask,
        sift=sift,
        fixed_angle=cfg.fixed_angle,
    )

    raw_count = 0
    after_cap_count = 0
    kept_in_page = 0
    rejected_border = 0
    rejected_low_fg = 0
    rejected_dedup = 0

    raw_list: List[np.ndarray] = []
    page_descriptor_rows: List[int] = []
    local_keypoint_indices: List[int] = []
    kp_x: List[float] = []
    kp_y: List[float] = []
    kp_size: List[float] = []
    kp_angle: List[float] = []
    kp_response: List[float] = []
    foreground_ratio: List[float] = []
    patch_names: List[str] = []
    patch_paths: List[str] = []

    if descriptors is not None and len(keypoints) > 0:
        raw_count = len(keypoints)

        keypoints, descriptors = sort_keypoints_by_response(
            keypoints=keypoints,
            descriptors=descriptors,
            max_keypoints_per_page=cfg.max_keypoints_per_page,
        )

        after_cap_count = len(keypoints)
        keypoints, descriptors, rejected_location_dedup = dedup_keypoints_by_location(
            keypoints=keypoints,
            descriptors=descriptors,
            round_decimals=cfg.rsift_dedup_location_round_decimals,
        )
        rejected_dedup += int(rejected_location_dedup)

        dedup_seen = set()

        for local_idx, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            patch = crop_patch_centered(
                img=img,
                x=kp.pt[0],
                y=kp.pt[1],
                patch_size=cfg.patch_size,
            )

            if patch is None:
                rejected_border += 1
                continue

            fg_ratio = patch_foreground_ratio(patch, cfg.foreground_threshold)
            if fg_ratio < cfg.min_foreground_ratio_in_patch:
                rejected_low_fg += 1
                continue

            center_ink_ratio = patch_center_ink_ratio(
                patch=patch,
                foreground_threshold=cfg.foreground_threshold,
                center_window=cfg.rsift_center_window,
            )
            if center_ink_ratio < cfg.rsift_min_center_ink_ratio:
                rejected_low_fg += 1
                continue

            if cfg.dedup_patches_per_page:
                patch_hash = patch_sha1(patch)
                if patch_hash in dedup_seen:
                    rejected_dedup += 1
                    continue
                dedup_seen.add(patch_hash)

            patch_filename = f"{image_id}_w{writer_id}_p{local_idx:05d}{cfg.patch_image_ext}"
            patch_path = patches_dir / patch_filename

            if cfg.save_patch_images:
                ok = cv2.imwrite(str(patch_path), patch)
                if not ok:
                    raise IOError(f"Failed to save patch image to {patch_path}")
                patch_path_str = str(patch_path)
            else:
                patch_path_str = ""

            page_row = len(raw_list)

            raw_list.append(desc.astype(np.float32))
            page_descriptor_rows.append(page_row)
            local_keypoint_indices.append(int(local_idx))
            kp_x.append(float(kp.pt[0]))
            kp_y.append(float(kp.pt[1]))
            kp_size.append(float(kp.size))
            kp_angle.append(float(kp.angle))
            kp_response.append(float(kp.response))
            foreground_ratio.append(float(fg_ratio))
            patch_names.append(patch_filename)
            patch_paths.append(patch_path_str)

            kept_in_page += 1

    if len(raw_list) > 0:
        raw_descriptors = np.stack(raw_list, axis=0).astype(np.float32)
        rootsift = rootsift_descriptors(raw_descriptors)
    else:
        raw_descriptors = np.empty((0, 128), dtype=np.float32)
        rootsift = np.empty((0, 128), dtype=np.float32)

    stats = PageExtractionStats(
        image_id=image_id,
        filename=filename,
        image_path=image_path,
        writer_id=writer_id,
        split=split_name,
        total_keypoints_raw=int(raw_count),
        total_keypoints_after_cap=int(after_cap_count),
        total_keypoints_kept=int(kept_in_page),
        rejected_border=int(rejected_border),
        rejected_low_foreground=int(rejected_low_fg),
        rejected_dedup=int(rejected_dedup),
        page_npz_path=str(page_npz_path),
    )

    atomic_savez_compressed(
        page_npz_path,
        raw_descriptors=raw_descriptors,
        rootsift_descriptors=rootsift,
        page_descriptor_row=np.asarray(page_descriptor_rows, dtype=np.int64),
        local_keypoint_index=np.asarray(local_keypoint_indices, dtype=np.int64),
        kp_x=np.asarray(kp_x, dtype=np.float32),
        kp_y=np.asarray(kp_y, dtype=np.float32),
        kp_size=np.asarray(kp_size, dtype=np.float32),
        kp_angle=np.asarray(kp_angle, dtype=np.float32),
        kp_response=np.asarray(kp_response, dtype=np.float32),
        foreground_ratio=np.asarray(foreground_ratio, dtype=np.float32),
        patch_name=np.asarray(patch_names, dtype="<U256"),
        patch_path=np.asarray(patch_paths, dtype="<U1024"),
        image_id=np.asarray(image_id),
        filename=np.asarray(filename),
        image_path=np.asarray(image_path),
        writer_id=np.asarray(writer_id, dtype=np.int64),
        split=np.asarray(split_name),
        stats_json=np.asarray(json.dumps(asdict(stats), ensure_ascii=False)),
    )

    return "processed", True, stats


def write_metadata_rows_from_page(
    writer: csv.DictWriter,
    data: Any,
    global_start_row: int,
) -> int:
    """Append metadata rows from one page npz and return number of rows written."""
    n = int(data["raw_descriptors"].shape[0])

    image_id = str(data["image_id"].item())
    filename = str(data["filename"].item())
    image_path = str(data["image_path"].item())
    writer_id = int(data["writer_id"].item())
    split = str(data["split"].item())

    page_descriptor_row = data["page_descriptor_row"]
    local_keypoint_index = data["local_keypoint_index"]
    kp_x = data["kp_x"]
    kp_y = data["kp_y"]
    kp_size = data["kp_size"]
    kp_angle = data["kp_angle"]
    kp_response = data["kp_response"]
    foreground_ratio = data["foreground_ratio"]
    patch_name = data["patch_name"]
    patch_path = data["patch_path"]

    for i in range(n):
        writer.writerow(
            {
                "descriptor_row": int(global_start_row + i),
                "page_descriptor_row": int(page_descriptor_row[i]),
                "split": split,
                "writer_id": writer_id,
                "image_id": image_id,
                "filename": filename,
                "image_path": image_path,
                "patch_name": str(patch_name[i]),
                "patch_path": str(patch_path[i]),
                "local_keypoint_index": int(local_keypoint_index[i]),
                "kp_x": float(kp_x[i]),
                "kp_y": float(kp_y[i]),
                "kp_size": float(kp_size[i]),
                "kp_angle": float(kp_angle[i]),
                "kp_response": float(kp_response[i]),
                "foreground_ratio": float(foreground_ratio[i]),
            }
        )

    return n


def aggregate_split_outputs(
    df: pd.DataFrame,
    split: str,
    output_root: Path,
    overwrite_aggregate: bool,
) -> Dict[str, Any]:
    """Aggregate per-page checkpoints into split-level .npy and .csv outputs."""
    split_root = output_root / split
    pages_dir = split_root / "pages"

    patch_metadata_path = split_root / f"{split}_patch_metadata.csv"
    page_stats_path = split_root / f"{split}_page_stats.csv"
    raw_desc_path = split_root / f"{split}_sift_raw_descriptors.npy"
    rootsift_desc_path = split_root / f"{split}_rootsift_descriptors.npy"
    split_summary_path = split_root / f"{split}_extraction_summary.json"

    tmp_metadata_path = patch_metadata_path.with_name(patch_metadata_path.name + ".tmp")
    tmp_page_stats_path = page_stats_path.with_name(page_stats_path.name + ".tmp")
    tmp_raw_desc_path = raw_desc_path.with_name(raw_desc_path.name + ".tmp")
    tmp_rootsift_desc_path = rootsift_desc_path.with_name(rootsift_desc_path.name + ".tmp")

    if (
        patch_metadata_path.exists()
        and page_stats_path.exists()
        and raw_desc_path.exists()
        and rootsift_desc_path.exists()
        and not overwrite_aggregate
    ):
        print(f"[AGG] {split}: aggregate outputs already exist; skipping aggregation.")
        with split_summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    page_paths: List[Path] = []
    stats_rows: List[Dict[str, Any]] = []
    total_rows = 0

    for row in df.itertuples(index=False):
        image_id = str(row.image_id)
        page_npz_path = pages_dir / f"{image_id}.npz"

        if not page_npz_is_valid(page_npz_path):
            raise FileNotFoundError(
                f"Missing or invalid page checkpoint for {split}: {page_npz_path}"
            )

        with np.load(page_npz_path, allow_pickle=False) as data:
            n = int(data["raw_descriptors"].shape[0])
            stats = json.loads(str(data["stats_json"].item()))

        page_paths.append(page_npz_path)
        stats_rows.append(stats)
        total_rows += n

    print(f"[AGG] {split}: pages={len(page_paths)}, descriptor_rows={total_rows}")

    raw_mmap = np.lib.format.open_memmap(
        tmp_raw_desc_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, 128),
    )
    root_mmap = np.lib.format.open_memmap(
        tmp_rootsift_desc_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, 128),
    )

    if tmp_metadata_path.exists():
        tmp_metadata_path.unlink()

    global_row = 0
    with tmp_metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()

        for page_npz_path in tqdm(page_paths, desc=f"Aggregate {split}"):
            with np.load(page_npz_path, allow_pickle=False) as data:
                raw = data["raw_descriptors"]
                root = data["rootsift_descriptors"]
                n = int(raw.shape[0])

                raw_mmap[global_row : global_row + n] = raw
                root_mmap[global_row : global_row + n] = root

                written = write_metadata_rows_from_page(
                    writer=writer,
                    data=data,
                    global_start_row=global_row,
                )

                if written != n:
                    raise ValueError(
                        f"Metadata rows written ({written}) do not match descriptors ({n}) "
                        f"for {page_npz_path}"
                    )

                global_row += n

    raw_mmap.flush()
    root_mmap.flush()
    del raw_mmap
    del root_mmap

    page_stats_df = pd.DataFrame(stats_rows)
    page_stats_df.to_csv(tmp_page_stats_path, index=False)

    os.replace(tmp_metadata_path, patch_metadata_path)
    os.replace(tmp_page_stats_path, page_stats_path)
    os.replace(tmp_raw_desc_path, raw_desc_path)
    os.replace(tmp_rootsift_desc_path, rootsift_desc_path)

    patches_per_page = page_stats_df["total_keypoints_kept"] if len(page_stats_df) else pd.Series(dtype=int)

    summary = {
        "split": split,
        "num_pages": int(len(df)),
        "num_pages_with_any_patch": int((patches_per_page > 0).sum()) if len(page_stats_df) else 0,
        "num_patches": int(total_rows),
        "raw_keypoints_total": int(page_stats_df["total_keypoints_raw"].sum()) if len(page_stats_df) else 0,
        "after_cap_keypoints_total": int(page_stats_df["total_keypoints_after_cap"].sum()) if len(page_stats_df) else 0,
        "kept_keypoints_total": int(page_stats_df["total_keypoints_kept"].sum()) if len(page_stats_df) else 0,
        "rejected_border_total": int(page_stats_df["rejected_border"].sum()) if len(page_stats_df) else 0,
        "rejected_low_foreground_total": int(page_stats_df["rejected_low_foreground"].sum()) if len(page_stats_df) else 0,
        "rejected_dedup_total": int(page_stats_df["rejected_dedup"].sum()) if len(page_stats_df) else 0,
        "mean_patches_per_page": float(total_rows / len(df)) if len(df) > 0 else 0.0,
        "min_patches_per_page": int(patches_per_page.min()) if len(patches_per_page) else 0,
        "max_patches_per_page": int(patches_per_page.max()) if len(patches_per_page) else 0,
        "raw_descriptor_shape": [int(total_rows), 128],
        "rootsift_descriptor_shape": [int(total_rows), 128],
        "outputs": {
            "patch_metadata": str(patch_metadata_path),
            "page_stats": str(page_stats_path),
            "sift_raw_descriptors": str(raw_desc_path),
            "rootsift_descriptors": str(rootsift_desc_path),
            "pages_dir": str(pages_dir),
        },
    }

    atomic_write_json(split_summary_path, summary)

    print(f"[OK] {split} aggregation complete")
    print(f"[OUT] {patch_metadata_path}")
    print(f"[OUT] {page_stats_path}")
    print(f"[OUT] {raw_desc_path}")
    print(f"[OUT] {rootsift_desc_path}")

    return summary


def extract_split_resumable(
    df: pd.DataFrame,
    split: str,
    cfg: ExtractionConfig,
    output_root: Path,
    overwrite_pages: bool,
    overwrite_aggregate: bool,
    aggregate: bool,
) -> Dict[str, Any]:
    """Run resumable page-level extraction and optional aggregation for one split."""
    split_root = output_root / split
    pages_dir = split_root / "pages"
    patches_dir = split_root / "patches"
    progress_path = split_root / f"{split}_progress.json"

    split_root.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)

    sift = create_sift_detector(cfg)

    processed = 0
    skipped = 0
    failed = 0

    last_stats: Optional[PageExtractionStats] = None

    for page_idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc=f"Extract {split}")):
        try:
            status, did_process, stats = extract_one_page(
                row=row,
                split=split,
                cfg=cfg,
                sift=sift,
                pages_dir=pages_dir,
                patches_dir=patches_dir,
                overwrite_pages=overwrite_pages,
            )

            if status == "processed":
                processed += 1
            elif status == "skipped":
                skipped += 1

            last_stats = stats

        except Exception as exc:
            failed += 1
            progress = {
                "split": split,
                "status": "failed",
                "page_idx": int(page_idx),
                "processed_pages_this_run": int(processed),
                "skipped_pages_this_run": int(skipped),
                "failed_pages_this_run": int(failed),
                "error": repr(exc),
            }
            atomic_write_json(progress_path, progress)
            raise

        if (page_idx + 1) % 50 == 0 or (page_idx + 1) == len(df):
            progress = {
                "split": split,
                "status": "running",
                "pages_seen": int(page_idx + 1),
                "total_pages": int(len(df)),
                "processed_pages_this_run": int(processed),
                "skipped_pages_this_run": int(skipped),
                "failed_pages_this_run": int(failed),
                "last_image_id": last_stats.image_id if last_stats else None,
            }
            atomic_write_json(progress_path, progress)

            print(
                f"[{split}] pages_seen={page_idx + 1}/{len(df)} | "
                f"processed={processed} | skipped={skipped} | failed={failed}"
            )

    progress = {
        "split": split,
        "status": "page_extraction_complete",
        "total_pages": int(len(df)),
        "processed_pages_this_run": int(processed),
        "skipped_pages_this_run": int(skipped),
        "failed_pages_this_run": int(failed),
        "pages_dir": str(pages_dir),
    }
    atomic_write_json(progress_path, progress)

    if not aggregate:
        return {
            "split": split,
            "status": "page_extraction_complete_without_aggregation",
            "progress": progress,
        }

    summary = aggregate_split_outputs(
        df=df,
        split=split,
        output_root=output_root,
        overwrite_aggregate=overwrite_aggregate,
    )

    progress["status"] = "complete_with_aggregation"
    atomic_write_json(progress_path, progress)

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 01R: resumable R-SIFT-like/RootSIFT/patch extraction for strict Christlein 2017 reproduction branch."
    )

    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Dataset split to process.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on a small deterministic subset.",
    )

    parser.add_argument(
        "--max-pages-per-split",
        type=int,
        default=3,
        help="Maximum pages per split when --smoke-test is enabled.",
    )

    parser.add_argument(
        "--max-keypoints-per-page",
        type=int,
        default=0,
        help="Maximum SIFT keypoints kept per page after response sorting. Use 0 for no cap.",
    )

    parser.add_argument(
        "--fixed-angle",
        action="store_true",
        help="Force keypoint angle to 0. Disabled by default for faithful mode.",
    )

    parser.add_argument(
        "--save-patch-images",
        action="store_true",
        help="Save patch image files. Disabled by default for no-cap faithful extraction.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root directory.",
    )

    parser.add_argument(
        "--overwrite-pages",
        action="store_true",
        help="Recompute page-level checkpoints even if valid .npz files already exist.",
    )

    parser.add_argument(
        "--overwrite-aggregate",
        action="store_true",
        help="Overwrite split-level aggregate .npy/.csv outputs.",
    )

    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Only write page-level checkpoints; do not build split-level aggregate files.",
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip extraction and aggregate existing valid page-level checkpoints.",
    )

    return parser.parse_args()


def main() -> None:
    """Run resumable SIFT / RootSIFT / patch extraction."""
    args = parse_args()

    cfg = ExtractionConfig(
        max_keypoints_per_page=int(args.max_keypoints_per_page),
        fixed_angle=bool(args.fixed_angle),
        save_patch_images=bool(args.save_patch_images),
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("[CONFIG]")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    print(f"[OUTPUT_ROOT] {output_root}")

    all_summary: Dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "input_manifests": {
            "train": str(TRAIN_MANIFEST),
            "test": str(TEST_MANIFEST),
        },
        "output_root": str(output_root),
        "stage": "01R_rsift_rootsift_patches",
        "methodological_status": "python_opencv_rsift_emulation_dark_ink_filtering_duplicate_location_removal",
        "canonical_image": "dark_ink_on_bright_background",
        "sift_detection_mask_used": False,
        "rsift_filtering": {
            "patch_ink_ratio_field": "foreground_ratio",
            "min_patch_ink_ratio": float(cfg.min_foreground_ratio_in_patch),
            "center_window": int(cfg.rsift_center_window),
            "min_center_ink_ratio": float(cfg.rsift_min_center_ink_ratio),
            "dedup_location_round_decimals": int(cfg.rsift_dedup_location_round_decimals),
        },
        "config": asdict(cfg),
        "smoke_test": bool(args.smoke_test),
        "max_pages_per_split": int(args.max_pages_per_split) if args.smoke_test else None,
        "aggregate": not bool(args.no_aggregate),
        "aggregate_only": bool(args.aggregate_only),
        "splits": {},
    }

    splits_to_run = []
    if args.split in ["train", "both"]:
        splits_to_run.append(("train", TRAIN_MANIFEST))
    if args.split in ["test", "both"]:
        splits_to_run.append(("test", TEST_MANIFEST))

    for split, manifest_path in splits_to_run:
        df = load_manifest(manifest_path)
        if args.smoke_test:
            df = apply_smoke_subset(df, split, args.max_pages_per_split)

        if args.aggregate_only:
            summary = aggregate_split_outputs(
                df=df,
                split=split,
                output_root=output_root,
                overwrite_aggregate=True,
            )
        else:
            summary = extract_split_resumable(
                df=df,
                split=split,
                cfg=cfg,
                output_root=output_root,
                overwrite_pages=bool(args.overwrite_pages),
                overwrite_aggregate=bool(args.overwrite_aggregate),
                aggregate=not bool(args.no_aggregate),
            )

        all_summary["splits"][split] = summary

    summary_path = output_root / "extraction_summary.json"
    atomic_write_json(summary_path, all_summary)

    print(f"[OK] Global extraction summary written to: {summary_path}")


if __name__ == "__main__":
    main()