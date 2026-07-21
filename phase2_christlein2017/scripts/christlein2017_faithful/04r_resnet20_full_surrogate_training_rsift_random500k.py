"""
Stage 04R: Strict-branch ResNet-20 surrogate training.

This is the strict-branch ResNet-20 / 64-D surrogate CNN training stage
using R-SIFT-derived surrogate labels.

Methodological target:
- Use Stage 03 accepted surrogate patches from TRAIN only.
- Labels are surrogate visual cluster labels, not writer labels.
- Model is ResNet-20 with 64-D penultimate descriptors.
- Optimizer is SGD with Nesterov momentum and weight decay.
- Validation uses held-out surrogate patches from TRAIN, not TEST.
- TEST data is not used in fitting/training.
- Checkpoint/resume is supported at epoch and post-train level.
- Images are canonicalized to dark ink on bright background before patch cropping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


cv2.setNumThreads(0)

PROJECT_ROOT = Path.cwd()
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from christlein2017_faithful.resnet20_pre_activation import (  # noqa: E402
    build_resnet20_64d,
    count_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--assign-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/surrogate_assignments_k5000_ratio09/train/pages",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/resnet20_surrogate_training",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="full_rsift_random500k_k5000_ratio09_resnet20_64d_seed123",
    )

    parser.add_argument("--num-classes", type=int, default=5000)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--lr-factor", type=float, default=0.1)
    parser.add_argument("--lr-patience", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=10)

    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--nesterov", action="store_true", default=True)
    parser.add_argument("--no-nesterov", action="store_false", dest="nesterov")

    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)

    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--max-image-cache", type=int, default=64)

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


def append_history_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_history_json(path: Path, history: List[Dict[str, Any]]) -> None:
    write_json(path, history)


def to_scalar_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        value = arr.item()
    elif arr.size == 1:
        value = arr.reshape(-1)[0].item()

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def find_key(npz, candidates: List[str]):
    keys = set(npz.files)
    for key in candidates:
        if key in keys:
            return key
    return None


def read_stage03_page(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        label_key = find_key(z, ["surrogate_label", "surrogate_labels", "labels", "cluster_label"])
        if label_key is None:
            raise KeyError(f"Could not find surrogate labels in {path}. Keys={z.files}")

        labels = np.asarray(z[label_key], dtype=np.int64)

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
        else:
            raise KeyError(f"Could not find keypoint coordinates in {path}. Keys={z.files}")

        image_path_key = find_key(z, ["image_path", "path", "filename_path"])
        if image_path_key is None:
            raise KeyError(f"Could not find image_path in {path}. Keys={z.files}")

        image_path = resolve_path(to_scalar_string(z[image_path_key]))

    if len(labels) != len(xs) or len(labels) != len(ys):
        raise ValueError(
            f"Length mismatch in {path}: labels={len(labels)}, xs={len(xs)}, ys={len(ys)}"
        )

    return {
        "page_id": path.stem,
        "image_path": image_path,
        "labels": labels,
        "xs": xs,
        "ys": ys,
    }


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


def canonical_dark_ink_image(img: np.ndarray) -> np.ndarray:
    """Return image in canonical dark-ink-on-bright-background form.

    Stage 01R uses the same convention before R-SIFT-like keypoint filtering
    and patch extraction. Stage 04R applies the same convention before
    re-cropping accepted surrogate patches for CNN training.
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


class LRUImageCache:
    """Small LRU cache for grayscale page images."""

    def __init__(self, max_items: int) -> None:
        self.max_items = int(max_items)
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)

        if key in self.cache:
            img = self.cache.pop(key)
            self.cache[key] = img
            return img

        img = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        img = canonical_dark_ink_image(img)

        self.cache[key] = img

        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

        return img


class Stage03PatchDataset(Dataset):
    """Dataset over selected Stage 03 accepted surrogate patches."""

    def __init__(
        self,
        assign_root: Path,
        sample_pairs: np.ndarray,
        patch_size: int,
        max_image_cache: int,
    ) -> None:
        self.assign_root = assign_root
        self.page_files = sorted(assign_root.glob("*.npz"))
        self.sample_pairs = np.asarray(sample_pairs, dtype=np.int64)
        self.patch_size = int(patch_size)

        if self.sample_pairs.ndim != 2 or self.sample_pairs.shape[1] != 2:
            raise ValueError(f"sample_pairs must have shape [N,2], got {self.sample_pairs.shape}")

        self.page_cache: Dict[int, Dict[str, Any]] = {}
        self.image_cache = LRUImageCache(max_items=max_image_cache)

    def _get_page_info(self, page_idx: int) -> Dict[str, Any]:
        page_idx = int(page_idx)
        if page_idx not in self.page_cache:
            self.page_cache[page_idx] = read_stage03_page(self.page_files[page_idx])
        return self.page_cache[page_idx]

    def __len__(self) -> int:
        return int(self.sample_pairs.shape[0])

    def __getitem__(self, idx: int):
        page_idx, local_idx = self.sample_pairs[idx]
        page_idx = int(page_idx)
        local_idx = int(local_idx)

        info = self._get_page_info(page_idx)
        img = self.image_cache.get(info["image_path"])

        x = float(info["xs"][local_idx])
        y = float(info["ys"][local_idx])
        label = int(info["labels"][local_idx])

        patch = crop_center_patch(img, x=x, y=y, patch_size=self.patch_size)

        # Keep the same preprocessing convention as validated in compact baseline:
        # grayscale input in [0, 1], single channel.
        tensor = torch.from_numpy(patch).float().div_(255.0).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


def build_or_load_full_split(
    assign_root: Path,
    out_root: Path,
    val_samples: int,
    seed: int,
    overwrite: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    split_path = out_root / "split_indices_full.npz"
    summary_path = out_root / "split_summary.json"

    if split_path.exists() and summary_path.exists() and not overwrite:
        print(f"[SPLIT] Loading existing full split: {split_path}")
        z = np.load(split_path)
        train_pairs = np.asarray(z["train_pairs"], dtype=np.int64)
        val_pairs = np.asarray(z["val_pairs"], dtype=np.int64)

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        return train_pairs, val_pairs, summary

    page_files = sorted(assign_root.glob("*.npz"))
    if not page_files:
        raise RuntimeError(f"No Stage 03 files found in {assign_root}")

    print("[SPLIT] Scanning all Stage 03 assignment pages")

    counts = []
    label_min = 10**9
    label_max = -1
    total_available = 0

    for page_idx, p in enumerate(page_files):
        info = read_stage03_page(p)
        labels = info["labels"]

        n = int(len(labels))
        counts.append(n)
        total_available += n

        label_min = min(label_min, int(labels.min()))
        label_max = max(label_max, int(labels.max()))

        if (page_idx + 1) % 100 == 0:
            print(f"[SPLIT] scanned pages={page_idx+1}/{len(page_files)} accepted={total_available}")

    if int(val_samples) <= 0:
        raise ValueError("val_samples must be positive.")

    if int(val_samples) >= total_available:
        raise ValueError(
            f"val_samples={val_samples} must be smaller than total_available={total_available}."
        )

    # Build exact non-duplicated full sample pair array.
    # Each row is [page_idx, local_idx].
    print("[SPLIT] Building full non-duplicated sample index")

    pairs = np.empty((total_available, 2), dtype=np.int64)
    offset = 0

    for page_idx, count in enumerate(counts):
        pairs[offset: offset + count, 0] = page_idx
        pairs[offset: offset + count, 1] = np.arange(count, dtype=np.int64)
        offset += count

    assert offset == total_available

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_available)

    val_n = int(val_samples)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    val_pairs = pairs[val_idx]
    train_pairs = pairs[train_idx]

    print("[SPLIT] Checking duplicate-free split")

    assert len(train_pairs) + len(val_pairs) == total_available
    assert len(val_pairs) == val_n

    tmp = split_path.with_name(split_path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        seed=np.array(seed, dtype=np.int64),
        total_available=np.array(total_available, dtype=np.int64),
    )
    os.replace(tmp, split_path)

    summary = {
        "stage": "04R_full_split_rsift_random500k",
        "assign_root": str(assign_root),
        "num_pages": int(len(page_files)),
        "total_available": int(total_available),
        "train_samples": int(len(train_pairs)),
        "val_samples": int(len(val_pairs)),
        "seed": int(seed),
        "label_min_full_scan": int(label_min),
        "label_max_full_scan": int(label_max),
        "split_path": str(split_path),
        "duplicate_free_by_construction": True,
        "test_data_used": False,
    }

    write_json(summary_path, summary)

    print("[SPLIT SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return train_pairs, val_pairs, summary


def make_loader(
    assign_root: Path,
    pairs: np.ndarray,
    patch_size: int,
    max_image_cache: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = Stage03PatchDataset(
        assign_root=assign_root,
        sample_pairs=pairs,
        patch_size=patch_size,
        max_image_cache=max_image_cache,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    epoch: int,
    max_batches: int,
    train: bool,
    num_classes: int,
) -> Dict[str, Any]:
    if train:
        model.train()
        prefix = "Train"
    else:
        model.eval()
        prefix = "Val"

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    num_batches = 0

    t0 = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches and max_batches > 0 and batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

            embedding, logits = model(x, return_embedding=True)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.inference_mode():
                embedding, logits = model(x, return_embedding=True)
                loss = criterion(logits, y)

        assert embedding.shape[1] == 64
        assert logits.shape[1] == num_classes
        assert y.min().item() >= 0
        assert y.max().item() < num_classes
        assert torch.isfinite(loss)

        pred = logits.argmax(dim=1)
        correct = int((pred == y).sum().detach().cpu())
        seen = int(y.numel())

        total_loss += float(loss.detach().cpu()) * seen
        total_correct += correct
        total_seen += seen
        num_batches += 1

        if batch_idx == 0 or (batch_idx + 1) % 500 == 0:
            avg_loss = total_loss / max(total_seen, 1)
            avg_acc = total_correct / max(total_seen, 1)
            print(
                f"[{prefix} epoch {epoch}] batch={batch_idx+1} "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f} seen={total_seen}"
            )

    if total_seen == 0:
        raise RuntimeError(f"{prefix}: no samples were processed.")

    acc = total_correct / total_seen
    return {
        "loss": total_loss / total_seen,
        "acc": acc,
        "error": 1.0 - acc,
        "seen": total_seen,
        "batches": num_batches,
        "seconds": time.time() - t0,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: List[Dict[str, Any]],
    config: Dict[str, Any],
    stage: str,
    best_val_error: float,
    best_val_loss: float,
    best_epoch: int,
    epochs_without_improvement: int,
    lr_reductions: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    torch.save(
        {
            "stage": stage,
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
            "best_val_error": float(best_val_error),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "epochs_without_improvement": int(epochs_without_improvement),
            "lr_reductions": int(lr_reductions),
        },
        tmp,
    )

    os.replace(tmp, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print(f"[RESUME] Loaded checkpoint: {path}")
    print(
        f"[RESUME] epoch={ckpt.get('epoch')} stage={ckpt.get('stage')} "
        f"history_rows={len(ckpt.get('history', []))}"
    )

    return ckpt


def reduce_lr_if_needed(
    optimizer: torch.optim.Optimizer,
    current_lr: float,
    factor: float,
    min_lr: float,
) -> Tuple[float, bool]:
    new_lr = max(current_lr * factor, min_lr)

    if new_lr >= current_lr:
        return current_lr, False

    for group in optimizer.param_groups:
        group["lr"] = new_lr

    return new_lr, True


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"

    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    assign_root = resolve_path(args.assign_root)
    out_root = resolve_path(args.output_root) / args.run_name
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "stage": "04R_resnet20_full_surrogate_training_rsift_random500k",
        "methodological_status": "strict_rsift_random500k_resnet20_full_surrogate_training_train_only",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "training_source": "TRAIN_ONLY_STAGE03_ACCEPTED_SURROGATE_PATCHES",
        "test_used_for_training": False,
        "canonical_image": "dark_ink_on_bright_background",
        "surrogate_labels": "KMeans ratio-test visual classes, not writer labels",
        "assign_root": str(assign_root),
        "output_root": str(out_root),
        "num_classes": int(args.num_classes),
        "embedding_dim": 64,
        "patch_size": int(args.patch_size),
        "val_samples": int(args.val_samples),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "device": str(device),
        "lr": float(args.lr),
        "min_lr": float(args.min_lr),
        "lr_factor": float(args.lr_factor),
        "lr_patience": int(args.lr_patience),
        "early_stop_patience": int(args.early_stop_patience),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "nesterov": bool(args.nesterov),
        "max_train_batches": int(args.max_train_batches),
        "max_val_batches": int(args.max_val_batches),
        "max_image_cache": int(args.max_image_cache),
        "test_data_used": False,
    }

    write_json(out_root / "run_config.json", config)

    train_pairs, val_pairs, split_summary = build_or_load_full_split(
        assign_root=assign_root,
        out_root=out_root,
        val_samples=int(args.val_samples),
        seed=int(args.seed),
        overwrite=bool(args.overwrite_split),
    )

    train_loader = make_loader(
        assign_root=assign_root,
        pairs=train_pairs,
        patch_size=int(args.patch_size),
        max_image_cache=int(args.max_image_cache),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=True,
        device=device,
    )

    val_loader = make_loader(
        assign_root=assign_root,
        pairs=val_pairs,
        patch_size=int(args.patch_size),
        max_image_cache=int(args.max_image_cache),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        device=device,
    )

    model = build_resnet20_64d(
        num_classes=int(args.num_classes),
        in_channels=1,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(args.lr),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        nesterov=bool(args.nesterov),
    )

    history: List[Dict[str, Any]] = []
    start_epoch = 1
    resume_stage = ""

    best_val_error = math.inf
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    lr_reductions = 0

    latest_path = ckpt_dir / "latest.pt"
    post_train_latest_path = ckpt_dir / "post_train_latest.pt"

    resume_path = resolve_path(args.resume) if args.resume else None
    if resume_path is None and post_train_latest_path.exists():
        resume_path = post_train_latest_path
    elif resume_path is None and latest_path.exists():
        resume_path = latest_path

    if resume_path is not None and resume_path.exists():
        ckpt = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        loaded_epoch = int(ckpt.get("epoch", 0))
        history = list(ckpt.get("history", []))
        resume_stage = str(ckpt.get("stage", ""))

        best_val_error = float(ckpt.get("best_val_error", math.inf))
        best_val_loss = float(ckpt.get("best_val_loss", math.inf))
        best_epoch = int(ckpt.get("best_epoch", 0))
        epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        lr_reductions = int(ckpt.get("lr_reductions", 0))

        if resume_stage == "post_train":
            start_epoch = loaded_epoch
        else:
            start_epoch = loaded_epoch + 1

    print("[RUN]")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("[MODEL]")
    print(f"total_params={total_params:,} trainable_params={trainable_params:,}")
    print("[SPLIT]")
    print(json.dumps(split_summary, indent=2, ensure_ascii=False))
    print(
        f"[START] start_epoch={start_epoch} resume_stage={resume_stage} "
        f"best_epoch={best_epoch} best_val_error={best_val_error}"
    )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        epoch_t0 = time.time()

        if resume_stage == "post_train" and epoch == start_epoch:
            print(f"[EPOCH {epoch}] Resumed from post-train checkpoint; skipping train phase.")
            train_metrics = history[-1]["train_metrics"]
        else:
            train_metrics = run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_batches=int(args.max_train_batches),
                train=True,
                num_classes=int(args.num_classes),
            )

            post_train_row = {
                "epoch": int(epoch),
                "train_metrics": train_metrics,
                "validation_pending": True,
            }

            temp_history = history + [post_train_row]

            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}_post_train.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=temp_history,
                config=config,
                stage="post_train",
                best_val_error=best_val_error,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                lr_reductions=lr_reductions,
            )

            save_checkpoint(
                post_train_latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=temp_history,
                config=config,
                stage="post_train",
                best_val_error=best_val_error,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                lr_reductions=lr_reductions,
            )

            print(f"[POST_TRAIN_CHECKPOINT] epoch={epoch}")

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch=epoch,
            max_batches=int(args.max_val_batches),
            train=False,
            num_classes=int(args.num_classes),
        )

        current_lr = float(optimizer.param_groups[0]["lr"])

        improved = bool(val_metrics["error"] < best_val_error)
        lr_reduced = False
        early_stop = False

        if improved:
            best_val_error = float(val_metrics["error"])
            best_val_loss = float(val_metrics["loss"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if improved:
            save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                config=config,
                stage="best",
                best_val_error=best_val_error,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                lr_reductions=lr_reductions,
            )
            print(f"[BEST] epoch={epoch} val_error={best_val_error:.6f} val_loss={best_val_loss:.6f}")

        if (not improved) and epochs_without_improvement >= int(args.lr_patience):
            new_lr, lr_reduced = reduce_lr_if_needed(
                optimizer=optimizer,
                current_lr=current_lr,
                factor=float(args.lr_factor),
                min_lr=float(args.min_lr),
            )

            if lr_reduced:
                lr_reductions += 1
                epochs_without_improvement = 0
                print(f"[LR] Reduced learning rate from {current_lr} to {new_lr}")
                current_lr = new_lr

        if epochs_without_improvement >= int(args.early_stop_patience):
            early_stop = True

        row = {
            "epoch": int(epoch),
            "epoch_seconds": time.time() - epoch_t0,
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "train_error": float(train_metrics["error"]),
            "train_seen": int(train_metrics["seen"]),
            "train_batches": int(train_metrics["batches"]),
            "train_seconds": float(train_metrics["seconds"]),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
            "val_error": float(val_metrics["error"]),
            "val_seen": int(val_metrics["seen"]),
            "val_batches": int(val_metrics["batches"]),
            "val_seconds": float(val_metrics["seconds"]),
            "lr": float(current_lr),
            "improved": bool(improved),
            "best_epoch": int(best_epoch),
            "best_val_error": float(best_val_error),
            "best_val_loss": float(best_val_loss),
            "epochs_without_improvement": int(epochs_without_improvement),
            "lr_reductions": int(lr_reductions),
            "early_stop": bool(early_stop),
        }

        history.append(row)

        save_history_json(out_root / "history.json", history)
        append_history_csv(out_root / "history.csv", row)

        save_checkpoint(
            ckpt_dir / f"epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            config=config,
            stage="epoch_complete",
            best_val_error=best_val_error,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            lr_reductions=lr_reductions,
        )

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            config=config,
            stage="epoch_complete",
            best_val_error=best_val_error,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            lr_reductions=lr_reductions,
        )

        if post_train_latest_path.exists():
            post_train_latest_path.unlink()

        summary = {
            "stage": "04R_resnet20_full_surrogate_training_rsift_random500k",
            "methodological_status": "strict_rsift_random500k_resnet20_full_surrogate_training_train_only",
            "config": config,
            "split_summary": split_summary,
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "history": history,
            "latest_epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val_error": float(best_val_error),
            "best_val_loss": float(best_val_loss),
            "lr_reductions": int(lr_reductions),
            "completed": False,
            "early_stopped": bool(early_stop),
        }

        write_json(out_root / "training_summary.json", summary)

        print("[EPOCH SUMMARY]")
        print(json.dumps(row, indent=2, ensure_ascii=False))

        resume_stage = ""

        if early_stop:
            print(f"[EARLY STOP] epoch={epoch} no improvement for {epochs_without_improvement} epochs")
            break

    final_summary = {
        "stage": "04R_resnet20_full_surrogate_training_rsift_random500k",
        "methodological_status": "strict_rsift_random500k_resnet20_full_surrogate_training_train_only",
        "config": config,
        "split_summary": split_summary,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "history": history,
        "latest_epoch": int(history[-1]["epoch"]) if history else 0,
        "best_epoch": int(best_epoch),
        "best_val_error": float(best_val_error),
        "best_val_loss": float(best_val_loss),
        "lr_reductions": int(lr_reductions),
        "completed": True,
        "early_stopped": bool(history[-1].get("early_stop", False)) if history else False,
        "passed": True,
    }

    write_json(out_root / "training_summary.json", final_summary)

    print("[OK] Stage 04R strict-branch ResNet-20 surrogate training completed.")
    print(f"[OUT] {out_root}")


if __name__ == "__main__":
    main()
