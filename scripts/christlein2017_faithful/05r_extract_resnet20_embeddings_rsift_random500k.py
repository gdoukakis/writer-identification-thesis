"""
Stage 05R: Extract strict-branch ResNet-20 64-D local embeddings for TRAIN and TEST pages.

Input:
- Stage 01 SIFT keypoint page files.
- Best Stage 04R ResNet-20 checkpoint.

Output:
- Per-page .npz files containing 64-D embeddings.
- Split-level summaries.

Methodological status:
- Strict-branch ResNet-20 / 64-D local descriptor extraction.
- TRAIN and TEST are processed independently.
- No fitting is performed here.
- Images are canonicalized to dark ink on bright background before patch cropping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


cv2.setNumThreads(0)

PROJECT_ROOT = Path.cwd()
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from christlein2017_faithful.resnet20_pre_activation import build_resnet20_64d  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage01-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/rsift_rootsift_patches",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/resnet20_surrogate_training/full_rsift_random500k_k5000_ratio09_resnet20_64d_seed123/checkpoints/best.pt",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/resnet20_embeddings",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="resnet20_64d_epoch022_best",
    )

    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--num-classes", type=int, default=5000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")

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


def to_scalar_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        value = arr.item()
    elif arr.size == 1:
        value = arr.reshape(-1)[0].item()

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def find_key(npz, candidates: List[str]) -> Optional[str]:
    keys = set(npz.files)
    for key in candidates:
        if key in keys:
            return key
    return None


def canonical_dark_ink_image(img: np.ndarray) -> np.ndarray:
    """Return image in canonical dark-ink-on-bright-background form.

    Stage 01R and Stage 04R use this convention. Stage 05R must use the same
    convention before re-cropping patches for 64-D embedding extraction.
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


def crop_center_patch(img: np.ndarray, x: float, y: float, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    padded = np.pad(img, ((half, half), (half, half)), mode="constant", constant_values=255)

    cx = int(round(float(x))) + half
    cy = int(round(float(y))) + half

    patch = padded[cy - half: cy + half, cx - half: cx + half]

    if patch.shape != (patch_size, patch_size):
        fixed = np.full((patch_size, patch_size), 255, dtype=np.uint8)
        h = min(patch.shape[0], patch_size)
        w = min(patch.shape[1], patch_size)
        fixed[:h, :w] = patch[:h, :w]
        patch = fixed

    return patch


def read_stage01_page(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        image_path_key = find_key(
            z,
            [
                "image_path",
                "path",
                "filename_path",
                "page_path",
                "source_image_path",
                "input_image_path",
            ],
        )
        if image_path_key is None:
            raise KeyError(f"Could not find image path in {path}. Keys={z.files}")

        image_path = resolve_path(to_scalar_string(z[image_path_key]))

        x_key = find_key(z, ["kp_x", "x", "xs", "keypoint_x", "keypoints_x"])
        y_key = find_key(z, ["kp_y", "y", "ys", "keypoint_y", "keypoints_y"])

        if x_key is not None and y_key is not None:
            xs = np.asarray(z[x_key], dtype=np.float32)
            ys = np.asarray(z[y_key], dtype=np.float32)
        elif "keypoints" in z.files:
            kp = np.asarray(z["keypoints"])
            xs = kp[:, 0].astype(np.float32)
            ys = kp[:, 1].astype(np.float32)
        elif "kp_xy_size_angle_response_fg" in z.files:
            kp = np.asarray(z["kp_xy_size_angle_response_fg"])
            xs = kp[:, 0].astype(np.float32)
            ys = kp[:, 1].astype(np.float32)
        elif "kp_xy_size_angle_response" in z.files:
            kp = np.asarray(z["kp_xy_size_angle_response"])
            xs = kp[:, 0].astype(np.float32)
            ys = kp[:, 1].astype(np.float32)
        else:
            raise KeyError(f"Could not find keypoint coordinates in {path}. Keys={z.files}")

        page_id_key = find_key(z, ["page_id", "image_id", "id"])
        writer_id_key = find_key(z, ["writer_id", "writer", "label_writer"])

        page_id = to_scalar_string(z[page_id_key]) if page_id_key else path.stem
        writer_id = to_scalar_string(z[writer_id_key]) if writer_id_key else ""

    if len(xs) != len(ys):
        raise ValueError(f"Coordinate length mismatch in {path}: xs={len(xs)} ys={len(ys)}")

    return {
        "page_id": page_id,
        "writer_id": writer_id,
        "image_path": image_path,
        "xs": xs,
        "ys": ys,
    }


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_resnet20_64d(num_classes=num_classes, in_channels=1).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    model.load_state_dict(state, strict=True)
    model.eval()

    print("[MODEL]")
    print("checkpoint:", checkpoint_path)
    print("checkpoint_epoch:", ckpt.get("epoch", "NA") if isinstance(ckpt, dict) else "NA")
    print("best_epoch:", ckpt.get("best_epoch", "NA") if isinstance(ckpt, dict) else "NA")
    print("num_classes:", num_classes)

    return model


def extract_page_embeddings(
    model: torch.nn.Module,
    page_npz: Path,
    output_npz: Path,
    split: str,
    checkpoint_path: Path,
    batch_size: int,
    patch_size: int,
    embedding_dim: int,
    device: torch.device,
) -> Dict[str, Any]:
    info = read_stage01_page(page_npz)

    img = cv2.imread(str(info["image_path"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {info['image_path']}")

    img = canonical_dark_ink_image(img)

    xs = info["xs"]
    ys = info["ys"]
    n = len(xs)

    embeddings = np.empty((n, embedding_dim), dtype=np.float32)

    t0 = time.time()

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            patches = np.empty((end - start, 1, patch_size, patch_size), dtype=np.float32)

            for j, idx in enumerate(range(start, end)):
                patch = crop_center_patch(img, xs[idx], ys[idx], patch_size)
                patches[j, 0] = patch.astype(np.float32) / 255.0

            x = torch.from_numpy(patches).to(device, non_blocking=True)
            emb = model.extract_embeddings(x)

            if emb.shape[1] != embedding_dim:
                raise RuntimeError(f"Expected embedding_dim={embedding_dim}, got {emb.shape}")

            embeddings[start:end] = emb.detach().cpu().numpy().astype(np.float32)

    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"Non-finite embeddings detected in {page_npz}")

    output_npz.parent.mkdir(parents=True, exist_ok=True)

    tmp = output_npz.with_name(output_npz.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        embeddings=embeddings,
        xs=xs.astype(np.float32),
        ys=ys.astype(np.float32),
        page_id=np.array(info["page_id"]),
        writer_id=np.array(info["writer_id"]),
        image_path=np.array(str(info["image_path"])),
        source_npz=np.array(str(page_npz)),
        split=np.array(split),
        checkpoint_path=np.array(str(checkpoint_path)),
    )
    os.replace(tmp, output_npz)

    return {
        "page_file": str(page_npz),
        "output_file": str(output_npz),
        "page_id": info["page_id"],
        "writer_id": info["writer_id"],
        "image_path": str(info["image_path"]),
        "num_embeddings": int(n),
        "embedding_dim": int(embedding_dim),
        "seconds": time.time() - t0,
        "mean_sample": float(embeddings[: min(n, 1000)].mean()) if n else None,
        "std_sample": float(embeddings[: min(n, 1000)].std()) if n else None,
    }


def process_split(
    split: str,
    stage01_root: Path,
    out_root: Path,
    model: torch.nn.Module,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    in_dir = stage01_root / split / "pages"
    out_dir = out_root / split / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    page_files = sorted(in_dir.glob("*.npz"))
    if not page_files:
        raise RuntimeError(f"No page files found for split={split}: {in_dir}")

    split_summary_path = out_root / split / f"{split}_embedding_summary.json"
    progress_path = out_root / split / f"{split}_page_results.jsonl"

    total_embeddings = 0
    failed_pages = []
    processed_pages = 0
    skipped_pages = 0
    sample_means = []
    sample_stds = []

    t0 = time.time()

    print(f"[SPLIT {split.upper()}]")
    print("input_dir:", in_dir)
    print("output_dir:", out_dir)
    print("num_pages:", len(page_files))

    with progress_path.open("a", encoding="utf-8") as progress_f:
        for i, page_npz in enumerate(page_files, start=1):
            output_npz = out_dir / page_npz.name

            if output_npz.exists() and not args.overwrite:
                try:
                    with np.load(output_npz, allow_pickle=True) as z:
                        n = int(z["embeddings"].shape[0])
                        d = int(z["embeddings"].shape[1])
                    if d == int(args.embedding_dim):
                        total_embeddings += n
                        skipped_pages += 1
                        processed_pages += 1
                        if i == 1 or i % 100 == 0:
                            print(f"[{split}] skip existing {i}/{len(page_files)} total_embeddings={total_embeddings}")
                        continue
                except Exception:
                    pass

            try:
                result = extract_page_embeddings(
                    model=model,
                    page_npz=page_npz,
                    output_npz=output_npz,
                    split=split,
                    checkpoint_path=checkpoint_path,
                    batch_size=int(args.batch_size),
                    patch_size=int(args.patch_size),
                    embedding_dim=int(args.embedding_dim),
                    device=device,
                )

                total_embeddings += int(result["num_embeddings"])
                processed_pages += 1

                if result["mean_sample"] is not None:
                    sample_means.append(result["mean_sample"])
                if result["std_sample"] is not None:
                    sample_stds.append(result["std_sample"])

                progress_f.write(json.dumps({"status": "ok", **result}, ensure_ascii=False) + "\n")
                progress_f.flush()

                if i == 1 or i % 50 == 0:
                    print(
                        f"[{split}] page={i}/{len(page_files)} "
                        f"embeddings={result['num_embeddings']} "
                        f"total_embeddings={total_embeddings} "
                        f"seconds={result['seconds']:.2f}"
                    )

            except Exception as exc:
                failed = {
                    "page_file": str(page_npz),
                    "error": repr(exc),
                }
                failed_pages.append(failed)
                progress_f.write(json.dumps({"status": "failed", **failed}, ensure_ascii=False) + "\n")
                progress_f.flush()
                print(f"[FAILED] {page_npz}: {exc}")

    summary = {
        "stage": "05R_resnet20_embedding_extraction_rsift_random500k",
        "methodological_status": "strict_rsift_random500k_resnet20_64d_embedding_extraction_no_fitting",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "embedding_source": "Stage01R accepted R-SIFT keypoint locations re-cropped as canonical dark-ink patches",
        "checkpoint_source": "Stage04R best.pt",
        "test_used_for_fitting": False,
        "canonical_image": "dark_ink_on_bright_background",
        "split": split,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "num_page_files": int(len(page_files)),
        "processed_pages": int(processed_pages),
        "skipped_pages": int(skipped_pages),
        "failed_pages": failed_pages,
        "num_failed_pages": int(len(failed_pages)),
        "total_embeddings": int(total_embeddings),
        "embedding_dim": int(args.embedding_dim),
        "checkpoint_path": str(checkpoint_path),
        "seconds": time.time() - t0,
        "sample_mean_mean": float(np.mean(sample_means)) if sample_means else None,
        "sample_std_mean": float(np.mean(sample_stds)) if sample_stds else None,
        "passed": len(failed_pages) == 0,
    }

    write_json(split_summary_path, summary)
    print(f"[SPLIT SUMMARY {split.upper()}]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


def main() -> None:
    args = parse_args()

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"

    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    stage01_root = resolve_path(args.stage01_root)
    checkpoint_path = resolve_path(args.checkpoint)
    out_root = resolve_path(args.output_root) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    config = {
        "stage": "05R_resnet20_embedding_extraction_rsift_random500k",
        "methodological_status": "strict_rsift_random500k_resnet20_64d_embedding_extraction_no_fitting",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "embedding_source": "Stage01R accepted R-SIFT keypoint locations re-cropped as canonical dark-ink patches",
        "checkpoint_source": "Stage04R best.pt",
        "test_used_for_fitting": False,
        "canonical_image": "dark_ink_on_bright_background",
        "stage01_root": str(stage01_root),
        "checkpoint_path": str(checkpoint_path),
        "output_root": str(out_root),
        "splits": splits,
        "num_classes": int(args.num_classes),
        "embedding_dim": int(args.embedding_dim),
        "patch_size": int(args.patch_size),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "overwrite": bool(args.overwrite),
    }

    write_json(out_root / "run_config.json", config)

    model = load_model(
        checkpoint_path=checkpoint_path,
        num_classes=int(args.num_classes),
        device=device,
    )

    summaries = []
    for split in splits:
        summaries.append(
            process_split(
                split=split,
                stage01_root=stage01_root,
                out_root=out_root,
                model=model,
                checkpoint_path=checkpoint_path,
                args=args,
                device=device,
            )
        )

    final_summary = {
        "stage": "05R_resnet20_embedding_extraction_rsift_random500k",
        "methodological_status": "strict_rsift_random500k_resnet20_64d_embedding_extraction_no_fitting",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "embedding_source": "Stage01R accepted R-SIFT keypoint locations re-cropped as canonical dark-ink patches",
        "checkpoint_source": "Stage04R best.pt",
        "test_used_for_fitting": False,
        "canonical_image": "dark_ink_on_bright_background",
        "config": config,
        "split_summaries": summaries,
        "passed": all(s["passed"] for s in summaries),
    }

    write_json(out_root / "run_summary.json", final_summary)

    print("[OK] Stage 05R strict-branch ResNet-20 64-D embedding extraction completed.")
    print("[OUT]", out_root)


if __name__ == "__main__":
    main()
