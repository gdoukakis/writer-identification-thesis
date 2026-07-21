#!/usr/bin/env python3
"""
Stage 07R: Exact E-SVM-FE for strict ResNet-20 + global SSR+L2 m-VLAD + PCA640 descriptors.

This stage implements the feature-encoding interpretation of Exemplar-SVM:

For each query/page descriptor x_i:
- train a binary Linear SVM with x_i as the only positive sample
- use TRAIN descriptors as negatives
- extract the SVM weight vector w
- L2-normalize w
- use normalized w as the new E-SVM-FE representation

Retrieval is then performed by cosine similarity between the new E-SVM-FE
representations, not by LinearSVM decision_function scores.

This is intended to correct Stage 07, which was an Exemplar-SVM scoring/reranking
variant rather than exact E-SVM feature encoding.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--descriptor-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/mvlad_resnet20/resnet20_64d_mvlad_5xk64_global_ssr_l2_pca640",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/esvm_fe_resnet20_exact",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="resnet20_64d_mvlad_5xk64_global_ssr_l2_pca640_exact_esvmfe",
    )

    parser.add_argument(
        "--c-grid",
        type=str,
        default="0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0",
    )
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--top-k", type=str, default="1,5,10")

    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--fixed-c", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def resolve_path(path_like: Any) -> Path:
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def to_py_scalar(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return x


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32, copy=False)


def average_precision_from_flags(relevant_flags: np.ndarray) -> float:
    relevant_flags = relevant_flags.astype(bool, copy=False)
    num_relevant = int(relevant_flags.sum())

    if num_relevant == 0:
        return 0.0

    ranks = np.arange(1, len(relevant_flags) + 1, dtype=np.float64)
    cumsum_rel = np.cumsum(relevant_flags).astype(np.float64)
    precision_at_k = cumsum_rel / ranks

    return float((precision_at_k * relevant_flags).sum() / num_relevant)


def reciprocal_rank_from_flags(relevant_flags: np.ndarray) -> float:
    rel_positions = np.flatnonzero(relevant_flags)
    if len(rel_positions) == 0:
        return 0.0
    return float(1.0 / (int(rel_positions[0]) + 1))


def train_exemplar_svm_weight(
    positive: np.ndarray,
    negatives: np.ndarray,
    c_value: float,
    max_iter: int,
    tol: float,
    seed: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, bool]:
    """Train one Exemplar-SVM and return normalized weight vector w / ||w||."""
    x = np.vstack([positive.reshape(1, -1), negatives]).astype(np.float32, copy=False)

    y = np.empty(x.shape[0], dtype=np.int32)
    y[0] = 1
    y[1:] = -1

    model = LinearSVC(
        C=float(c_value),
        class_weight="balanced",
        dual=False,
        fit_intercept=True,
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(seed),
    )

    converged = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)

        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                converged = False
                break

    weight = model.coef_.reshape(-1).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(weight))

    if norm <= eps:
        feature = np.zeros_like(weight, dtype=np.float32)
    else:
        feature = (weight / norm).astype(np.float32, copy=False)

    return feature, converged


def compute_esvmfe_features(
    positives: np.ndarray,
    negatives: np.ndarray,
    c_value: float,
    max_iter: int,
    tol: float,
    seed: int,
    out_feature_path: Optional[Path] = None,
    overwrite: bool = False,
    progress_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute exact E-SVM-FE features for a set of positive descriptors."""
    if positives.ndim != 2:
        raise ValueError(f"Expected positives [N, D], got {positives.shape}")
    if negatives.ndim != 2:
        raise ValueError(f"Expected negatives [M, D], got {negatives.shape}")
    if positives.shape[1] != negatives.shape[1]:
        raise ValueError(
            f"Dim mismatch: positives={positives.shape}, negatives={negatives.shape}"
        )

    n, d = positives.shape
    num_convergence_warnings = 0

    if out_feature_path is not None:
        out_feature_path.parent.mkdir(parents=True, exist_ok=True)

    if out_feature_path is not None and out_feature_path.exists() and not overwrite:
        features = np.load(out_feature_path, mmap_mode="r+")
        if features.shape != (n, d):
            raise RuntimeError(
                f"Existing feature file has shape {features.shape}, expected {(n, d)}"
            )
        row_norms = np.linalg.norm(features, axis=1)
        completed_mask = row_norms > 0.5
        print(
            f"[RESUME] {out_feature_path} completed rows: "
            f"{int(completed_mask.sum())}/{n}"
        )
    else:
        if out_feature_path is not None:
            tmp_path = out_feature_path.with_name(out_feature_path.name + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            features = np.lib.format.open_memmap(
                tmp_path,
                mode="w+",
                dtype=np.float32,
                shape=(n, d),
            )
            features[:] = 0.0
            features.flush()
            os.replace(tmp_path, out_feature_path)
            features = np.load(out_feature_path, mmap_mode="r+")
        else:
            features = np.zeros((n, d), dtype=np.float32)

        completed_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        if completed_mask[i] and not overwrite:
            continue

        feature, converged = train_exemplar_svm_weight(
            positive=positives[i],
            negatives=negatives,
            c_value=c_value,
            max_iter=max_iter,
            tol=tol,
            seed=seed + i,
        )

        features[i] = feature

        if not converged:
            num_convergence_warnings += 1

        if out_feature_path is not None and (i == 0 or (i + 1) % 25 == 0):
            features.flush()

        if progress_path is not None and (i == 0 or (i + 1) % 50 == 0 or (i + 1) == n):
            row_norms_now = np.linalg.norm(features, axis=1)
            done_now = int((row_norms_now > 0.5).sum())

            write_json(
                progress_path,
                {
                    "status": "running",
                    "completed_rows": done_now,
                    "total_rows": int(n),
                    "c_value": float(c_value),
                    "num_convergence_warnings_this_run": int(num_convergence_warnings),
                    "feature_path": str(out_feature_path) if out_feature_path else None,
                },
            )

            print(
                f"[E-SVM-FE] {done_now}/{n} "
                f"C={c_value} convergence_warnings={num_convergence_warnings}"
            )

    if out_feature_path is not None:
        features.flush()

    features_arr = np.asarray(features, dtype=np.float32)

    norms = np.linalg.norm(features_arr, axis=1)
    completed = norms > 0.5

    summary = {
        "num_features": int(n),
        "feature_dim": int(d),
        "c_value": float(c_value),
        "num_convergence_warnings": int(num_convergence_warnings),
        "num_nonzero_features": int(completed.sum()),
        "feature_norm_mean": float(norms.mean()),
        "feature_norm_std": float(norms.std()),
        "feature_norm_min": float(norms.min()),
        "feature_norm_max": float(norms.max()),
        "features_finite": bool(np.isfinite(features_arr).all()),
        "passed": bool(completed.all() and np.isfinite(features_arr).all()),
    }

    return features_arr, summary


def evaluate_cosine_retrieval(
    features: np.ndarray,
    writer_ids: np.ndarray,
    top_ks: List[int],
    page_ids: Optional[np.ndarray] = None,
    image_paths: Optional[np.ndarray] = None,
    expected_relevant: Optional[int] = None,
    out_jsonl: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate leave-one-out retrieval with cosine/dot product on L2-normalized features."""
    features = l2_normalize_rows(features)

    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite E-SVM-FE features.")

    n = int(features.shape[0])
    scores = features @ features.T

    rows: List[Dict[str, Any]] = []
    top_hits = {k: [] for k in top_ks}
    aps = []
    rrs = []
    first_relevant_ranks = []

    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with (out_jsonl.open("w", encoding="utf-8") if out_jsonl is not None else open(os.devnull, "w")) as f:
        for q_idx in range(n):
            q_writer = writer_ids[q_idx]

            order = np.argsort(-scores[q_idx], kind="mergesort")
            ranking = order[order != q_idx]

            relevant_flags = writer_ids[ranking] == q_writer
            num_relevant = int(relevant_flags.sum())

            if expected_relevant is not None and num_relevant != expected_relevant:
                raise RuntimeError(
                    f"Query {q_idx}: expected {expected_relevant} relevant items, got {num_relevant}"
                )

            ap = average_precision_from_flags(relevant_flags)
            rr = reciprocal_rank_from_flags(relevant_flags)

            rel_positions = np.flatnonzero(relevant_flags)
            first_rank = int(rel_positions[0] + 1) if len(rel_positions) else -1

            row = {
                "query_index": int(q_idx),
                "query_page_id": str(page_ids[q_idx]) if page_ids is not None else str(q_idx),
                "query_writer_id": str(q_writer),
                "query_image_path": str(image_paths[q_idx]) if image_paths is not None else "",
                "num_relevant": int(num_relevant),
                "ap": float(ap),
                "reciprocal_rank": float(rr),
                "first_relevant_rank": int(first_rank),
                "top1_index": int(ranking[0]),
                "top1_writer_id": str(writer_ids[ranking[0]]),
                "top1_score": float(scores[q_idx, ranking[0]]),
            }

            if page_ids is not None:
                row["top1_page_id"] = str(page_ids[ranking[0]])

            for k in top_ks:
                hit = bool(relevant_flags[:k].any())
                top_hits[k].append(hit)
                row[f"top{k}_hit"] = hit

            rows.append(row)
            aps.append(ap)
            rrs.append(rr)
            first_relevant_ranks.append(first_rank)

            if out_jsonl is not None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if q_idx == 0 or (q_idx + 1) % 250 == 0:
                print(
                    f"[RETRIEVAL] q={q_idx + 1}/{n} "
                    f"AP={ap:.4f} RR={rr:.4f} first_rank={first_rank}"
                )

    aps_arr = np.asarray(aps, dtype=np.float64)
    rrs_arr = np.asarray(rrs, dtype=np.float64)
    first_ranks_arr = np.asarray(first_relevant_ranks, dtype=np.int64)

    metrics = {
        "num_queries": int(n),
        "mAP": float(aps_arr.mean()),
        "MRR": float(rrs_arr.mean()),
        "median_first_relevant_rank": float(np.median(first_ranks_arr)),
        "mean_first_relevant_rank": float(first_ranks_arr.mean()),
        "min_first_relevant_rank": int(first_ranks_arr.min()),
        "max_first_relevant_rank": int(first_ranks_arr.max()),
        "AP_min": float(aps_arr.min()),
        "AP_max": float(aps_arr.max()),
    }

    for k in top_ks:
        metrics[f"Top-{k}"] = float(np.mean(top_hits[k]))

    return metrics, rows


def make_writer_folds(writer_ids: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    writers = np.unique(writer_ids)
    rng = np.random.default_rng(seed)
    shuffled = writers.copy()
    rng.shuffle(shuffled)

    folds = np.array_split(shuffled, n_folds)
    return [np.asarray(fold) for fold in folds]


def select_c_by_exact_train_cv(
    train_desc: np.ndarray,
    train_writer_ids: np.ndarray,
    c_grid: List[float],
    top_ks: List[int],
    cv_folds: int,
    seed: int,
    max_iter: int,
    tol: float,
    out_root: Path,
    overwrite: bool,
) -> Tuple[float, Dict[str, Any]]:
    """Select C using exact E-SVM-FE on TRAIN-only writer-independent folds."""
    cv_summary_path = out_root / "cv_results_exact_esvmfe.json"

    if cv_summary_path.exists() and not overwrite:
        with cv_summary_path.open("r", encoding="utf-8") as f:
            cv_summary = json.load(f)
        print("[CV] Loading existing exact CV result:", cv_summary_path)
        return float(cv_summary["best_c"]), cv_summary

    folds = make_writer_folds(train_writer_ids, n_folds=cv_folds, seed=seed)

    all_results = []

    for c_value in c_grid:
        print(f"\n[CV EXACT E-SVM-FE] C={c_value}")
        fold_metrics = []

        for fold_idx, val_writers in enumerate(folds):
            val_mask = np.isin(train_writer_ids, val_writers)
            bg_mask = ~val_mask

            val_desc = train_desc[val_mask]
            val_writer_ids = train_writer_ids[val_mask]
            bg_desc = train_desc[bg_mask]

            unique_val, val_counts = np.unique(val_writer_ids, return_counts=True)
            expected_relevant = int(val_counts[0] - 1) if val_counts.min() == val_counts.max() else None

            print(
                f"[CV] C={c_value} fold={fold_idx + 1}/{cv_folds} "
                f"val_pages={len(val_desc)} bg_negatives={len(bg_desc)} "
                f"val_writers={len(unique_val)} expected_rel={expected_relevant}"
            )

            cv_feature_path = (
                out_root
                / "cv_features"
                / f"c_{str(c_value).replace('.', 'p')}_fold_{fold_idx}_features.npy"
            )
            cv_progress_path = (
                out_root
                / "cv_features"
                / f"c_{str(c_value).replace('.', 'p')}_fold_{fold_idx}_progress.json"
            )

            val_features, feature_summary = compute_esvmfe_features(
                positives=val_desc,
                negatives=bg_desc,
                c_value=float(c_value),
                max_iter=max_iter,
                tol=tol,
                seed=seed + fold_idx * 100000,
                out_feature_path=cv_feature_path,
                overwrite=overwrite,
                progress_path=cv_progress_path,
            )

            metrics, _ = evaluate_cosine_retrieval(
                features=val_features,
                writer_ids=val_writer_ids,
                top_ks=top_ks,
                expected_relevant=expected_relevant,
                out_jsonl=None,
            )

            metrics["fold"] = int(fold_idx)
            metrics["c_value"] = float(c_value)
            metrics["val_pages"] = int(len(val_desc))
            metrics["background_negatives"] = int(len(bg_desc))
            metrics["val_writers"] = int(len(unique_val))
            metrics["feature_summary"] = feature_summary

            fold_metrics.append(metrics)

            print(
                f"[CV FOLD RESULT] C={c_value} fold={fold_idx} "
                f"mAP={metrics['mAP']:.6f} Top1={metrics['Top-1']:.6f} "
                f"MRR={metrics['MRR']:.6f}"
            )

        mean_map = float(np.mean([m["mAP"] for m in fold_metrics]))
        mean_top1 = float(np.mean([m["Top-1"] for m in fold_metrics]))
        mean_mrr = float(np.mean([m["MRR"] for m in fold_metrics]))

        result = {
            "c_value": float(c_value),
            "mean_mAP": mean_map,
            "mean_Top-1": mean_top1,
            "mean_MRR": mean_mrr,
            "fold_metrics": fold_metrics,
        }

        all_results.append(result)

        print(
            f"[CV RESULT] C={c_value} "
            f"mean_mAP={mean_map:.6f} mean_Top1={mean_top1:.6f} mean_MRR={mean_mrr:.6f}"
        )

    best = sorted(
        all_results,
        key=lambda r: (r["mean_mAP"], r["mean_Top-1"], r["mean_MRR"], -r["c_value"]),
        reverse=True,
    )[0]

    cv_summary = {
        "stage": "07R_exact_esvmfe_train_only_c_selection_rsift_random500k_pca640",
        "methodological_status": "strict_rsift_random500k_exact_esvmfe_writer_independent_train_only_cv",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "descriptor_source": "Stage06R-B strict PCA640 whitened L2 descriptors",
        "cv_source": "TRAIN_ONLY",
        "c_grid": c_grid,
        "cv_folds": int(cv_folds),
        "seed": int(seed),
        "best_c": float(best["c_value"]),
        "best_mean_mAP": float(best["mean_mAP"]),
        "best_mean_Top-1": float(best["mean_Top-1"]),
        "best_mean_MRR": float(best["mean_MRR"]),
        "all_results": all_results,
        "test_data_used_for_c_selection": False,
        "ranking_method": "cosine_similarity_between_normalized_svm_weight_vectors",
        "passed": True,
    }

    write_json(cv_summary_path, cv_summary)

    csv_path = out_root / "cv_results_exact_esvmfe.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["c_value", "mean_mAP", "mean_Top-1", "mean_MRR"],
        )
        writer.writeheader()
        for row in all_results:
            writer.writerow({
                "c_value": row["c_value"],
                "mean_mAP": row["mean_mAP"],
                "mean_Top-1": row["mean_Top-1"],
                "mean_MRR": row["mean_MRR"],
            })

    return float(best["c_value"]), cv_summary


def save_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No rows to save.")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_py_scalar(v) for k, v in row.items()})


def main() -> None:
    args = parse_args()

    descriptor_root = resolve_path(args.descriptor_root)
    out_root = resolve_path(args.output_root) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    top_ks = [int(x.strip()) for x in args.top_k.split(",") if x.strip()]
    c_grid = [float(x.strip()) for x in args.c_grid.split(",") if x.strip()]

    train_desc_path = descriptor_root / "final_descriptors" / "train_pca640_whiten_l2.npy"
    test_desc_path = descriptor_root / "final_descriptors" / "test_pca640_whiten_l2.npy"

    train_writer_ids_path = descriptor_root / "arrays" / "train_writer_ids.npy"
    test_writer_ids_path = descriptor_root / "arrays" / "test_writer_ids.npy"
    test_page_ids_path = descriptor_root / "arrays" / "test_page_ids.npy"
    test_image_paths_path = descriptor_root / "arrays" / "test_image_paths.npy"

    print("[LOAD]")
    print("train descriptors:", train_desc_path)
    print("test descriptors :", test_desc_path)

    train_desc = np.load(train_desc_path).astype(np.float32, copy=False)
    test_desc = np.load(test_desc_path).astype(np.float32, copy=False)

    train_writer_ids = np.load(train_writer_ids_path, allow_pickle=True)
    test_writer_ids = np.load(test_writer_ids_path, allow_pickle=True)
    test_page_ids = np.load(test_page_ids_path, allow_pickle=True)

    if test_image_paths_path.exists():
        test_image_paths = np.load(test_image_paths_path, allow_pickle=True)
    else:
        test_image_paths = np.asarray([""] * len(test_page_ids))

    if not np.isfinite(train_desc).all():
        raise RuntimeError("Non-finite TRAIN descriptors.")
    if not np.isfinite(test_desc).all():
        raise RuntimeError("Non-finite TEST descriptors.")

    train_desc = l2_normalize_rows(train_desc)
    test_desc = l2_normalize_rows(test_desc)

    print("[SHAPES]")
    print("train:", train_desc.shape)
    print("test :", test_desc.shape)
    print("train writers:", len(np.unique(train_writer_ids)))
    print("test writers :", len(np.unique(test_writer_ids)))

    if args.skip_cv:
        if args.fixed_c is None:
            raise RuntimeError("--skip-cv requires --fixed-c")
        best_c = float(args.fixed_c)
        cv_summary = {
            "stage": "07R_exact_esvmfe_train_only_c_selection_rsift_random500k_pca640",
            "skipped": True,
            "best_c": best_c,
            "test_data_used_for_c_selection": False,
            "ranking_method": "cosine_similarity_between_normalized_svm_weight_vectors",
            "passed": True,
        }
        write_json(out_root / "cv_results_exact_esvmfe.json", cv_summary)
    else:
        best_c, cv_summary = select_c_by_exact_train_cv(
            train_desc=train_desc,
            train_writer_ids=train_writer_ids,
            c_grid=c_grid,
            top_ks=top_ks,
            cv_folds=int(args.cv_folds),
            seed=int(args.seed),
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            out_root=out_root,
            overwrite=bool(args.overwrite),
        )

    print("[BEST C]", best_c)

    test_feature_path = out_root / "test_exact_esvmfe_features.npy"
    test_progress_path = out_root / "test_exact_esvmfe_progress.json"

    test_features, test_feature_summary = compute_esvmfe_features(
        positives=test_desc,
        negatives=train_desc,
        c_value=float(best_c),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        seed=int(args.seed),
        out_feature_path=test_feature_path,
        overwrite=bool(args.overwrite),
        progress_path=test_progress_path,
    )

    unique_test, test_counts = np.unique(test_writer_ids, return_counts=True)
    expected_relevant = int(test_counts[0] - 1) if test_counts.min() == test_counts.max() else None

    jsonl_path = out_root / "per_query_results_exact_esvmfe.jsonl"

    metrics, rows = evaluate_cosine_retrieval(
        features=test_features,
        writer_ids=test_writer_ids,
        top_ks=top_ks,
        page_ids=test_page_ids,
        image_paths=test_image_paths,
        expected_relevant=expected_relevant,
        out_jsonl=jsonl_path,
    )

    csv_path = out_root / "per_query_results_exact_esvmfe.csv"
    save_rows_csv(csv_path, rows)

    feature_norms = np.linalg.norm(test_features, axis=1)

    summary = {
        "stage": "07R_exact_esvm_fe_resnet20_rsift_random500k_pca640",
        "methodological_status": "strict_rsift_random500k_exact_esvm_feature_encoding_with_normalized_svm_weights",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "descriptor_source": "Stage06R-B strict PCA640 whitened L2 descriptors",
        "esvm_positive_source": "each TEST descriptor as one positive exemplar",
        "esvm_negative_source": "TRAIN descriptors used as negatives",
        "retrieval_protocol": "ICDAR2017 Historical-WI TEST-only leave-one-out",
        "gallery": "all TEST E-SVM-FE representations except the query itself",
        "relevant_items": "other TEST pages of the same writer",
        "self_match_excluded": True,
        "test_descriptors_used_as_positive_exemplars": True,
        "descriptor_root": str(descriptor_root),
        "output_root": str(out_root),
        "train_descriptor_path": str(train_desc_path),
        "test_descriptor_path": str(test_desc_path),
        "best_c": float(best_c),
        "cv_summary_path": str(out_root / "cv_results_exact_esvmfe.json"),
        "test_esvmfe_feature_path": str(test_feature_path),
        "per_query_csv": str(csv_path),
        "per_query_jsonl": str(jsonl_path),
        "representation": "normalized_linear_svm_weight_vector_w_over_norm_w",
        "similarity": "cosine_similarity_between_exact_esvmfe_features",
        "self_match_removed": True,
        "train_descriptors_used_as_negatives": True,
        "test_data_used_for_c_selection": False,
        "ranking_uses_decision_function": False,
        "metrics": {
            **metrics,
            "descriptor_dim": int(test_features.shape[1]),
            "num_writers": int(len(unique_test)),
            "pages_per_writer_min": int(test_counts.min()),
            "pages_per_writer_max": int(test_counts.max()),
            "expected_relevant_per_query": int(expected_relevant) if expected_relevant is not None else None,
            "num_convergence_warnings": int(test_feature_summary["num_convergence_warnings"]),
        },
        "validation": {
            "train_descriptors_finite": bool(np.isfinite(train_desc).all()),
            "test_descriptors_finite": bool(np.isfinite(test_desc).all()),
            "test_esvmfe_features_finite": bool(np.isfinite(test_features).all()),
            "test_esvmfe_feature_norm_mean": float(feature_norms.mean()),
            "test_esvmfe_feature_norm_std": float(feature_norms.std()),
            "test_esvmfe_feature_norm_min": float(feature_norms.min()),
            "test_esvmfe_feature_norm_max": float(feature_norms.max()),
            "all_ap_leq_1": bool(all(float(r["ap"]) <= 1.0 + 1e-12 for r in rows)),
            "all_num_relevant_expected": bool(
                expected_relevant is not None and all(int(r["num_relevant"]) == expected_relevant for r in rows)
            ),
            "num_rows": int(len(rows)),
        },
        "test_feature_summary": test_feature_summary,
        "cv_summary": cv_summary,
        "passed": True,
    }

    write_json(out_root / "retrieval_summary_exact_esvmfe.json", summary)

    print("[METRICS]")
    print(json.dumps(summary["metrics"], indent=2))

    print("[VALIDATION]")
    print(json.dumps(summary["validation"], indent=2))

    print("[OK] Stage 07R strict exact E-SVM-FE retrieval evaluation completed.")
    print("[OUT]", out_root)


if __name__ == "__main__":
    main()