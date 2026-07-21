"""
Stage 06R-C: Retrieval evaluation for strict ResNet-20 + global SSR+L2 m-VLAD + PCA640 descriptors.

Protocol:
- TEST-only leave-one-out retrieval.
- Each TEST page is used as query.
- Gallery contains all TEST pages except the query page itself.
- Relevant items are the other pages of the same writer.
- ICDAR2017 Historical-WI TEST protocol: 720 writers x 5 pages = 3600 pages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


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
        default="outputs/faithful_christlein2017/strict_christlein2017_rsift_random500k/retrieval_eval_resnet20",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="resnet20_64d_mvlad_5xk64_global_ssr_l2_pca640_test_eval_full",
    )
    parser.add_argument("--descriptor-file", type=str, default="test_pca640_whiten_l2.npy")
    parser.add_argument("--top-k", type=str, default="1,5,10")
    parser.add_argument("--eps", type=float, default=1e-12)

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


def to_py_scalar(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return x


def average_precision_from_flags(relevant_flags: np.ndarray) -> float:
    """
    Compute AP over a ranked binary relevance vector.

    relevant_flags must exclude the query itself.
    """
    relevant_flags = relevant_flags.astype(bool, copy=False)
    num_relevant = int(relevant_flags.sum())

    if num_relevant == 0:
        return 0.0

    ranks = np.arange(1, len(relevant_flags) + 1, dtype=np.float64)
    cumsum_rel = np.cumsum(relevant_flags).astype(np.float64)
    precision_at_k = cumsum_rel / ranks

    ap = float((precision_at_k * relevant_flags).sum() / num_relevant)
    return ap


def reciprocal_rank_from_flags(relevant_flags: np.ndarray) -> float:
    rel_positions = np.flatnonzero(relevant_flags)
    if len(rel_positions) == 0:
        return 0.0
    return float(1.0 / (int(rel_positions[0]) + 1))


def main() -> None:
    args = parse_args()

    descriptor_root = resolve_path(args.descriptor_root)
    out_root = resolve_path(args.output_root) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    top_ks = [int(x.strip()) for x in args.top_k.split(",") if x.strip()]

    test_desc_path = descriptor_root / "final_descriptors" / args.descriptor_file
    test_writer_ids_path = descriptor_root / "arrays" / "test_writer_ids.npy"
    test_page_ids_path = descriptor_root / "arrays" / "test_page_ids.npy"
    test_image_paths_path = descriptor_root / "arrays" / "test_image_paths.npy"

    print("[LOAD]")
    print("descriptors:", test_desc_path)
    print("writer_ids :", test_writer_ids_path)
    print("page_ids   :", test_page_ids_path)

    descriptors = np.load(test_desc_path).astype(np.float32, copy=False)
    writer_ids = np.load(test_writer_ids_path, allow_pickle=True)
    page_ids = np.load(test_page_ids_path, allow_pickle=True)

    if test_image_paths_path.exists():
        image_paths = np.load(test_image_paths_path, allow_pickle=True)
    else:
        image_paths = np.asarray([""] * len(page_ids))

    if descriptors.ndim != 2:
        raise RuntimeError(f"Expected 2D descriptors, got {descriptors.shape}")

    n, dim = descriptors.shape

    if len(writer_ids) != n or len(page_ids) != n:
        raise RuntimeError(
            f"Metadata length mismatch: descriptors={n}, writers={len(writer_ids)}, pages={len(page_ids)}"
        )

    if not np.isfinite(descriptors).all():
        raise RuntimeError("Non-finite descriptor values detected.")

    norms = np.linalg.norm(descriptors, axis=1)
    print("[DESCRIPTOR STATS]")
    print("shape:", descriptors.shape)
    print("norm mean/std/min/max:", float(norms.mean()), float(norms.std()), float(norms.min()), float(norms.max()))

    unique_writers, writer_counts = np.unique(writer_ids, return_counts=True)
    print("[WRITER STATS]")
    print("num writers:", len(unique_writers))
    print("pages per writer min/max:", int(writer_counts.min()), int(writer_counts.max()))

    expected_relevant = int(writer_counts[0] - 1) if writer_counts.min() == writer_counts.max() else None

    if len(unique_writers) != 720:
        print("[WARN] Expected 720 TEST writers, got", len(unique_writers))

    if writer_counts.min() != 5 or writer_counts.max() != 5:
        print("[WARN] Expected exactly 5 pages per TEST writer.")

    print("[SIMILARITY]")
    sim = descriptors @ descriptors.T
    sim = sim.astype(np.float32, copy=False)

    # Self-similarity is excluded. We still physically remove the query index from
    # each ranking below to avoid counting the query as an additional relevant item.
    np.fill_diagonal(sim, -np.inf)

    per_query_rows: List[Dict[str, Any]] = []
    aps = []
    rrs = []
    first_relevant_ranks = []
    top_hits = {k: [] for k in top_ks}

    print("[EVAL]")
    for q_idx in range(n):
        scores = sim[q_idx]
        order = np.argsort(-scores, kind="mergesort")

        # Critical: physically remove self-match from the ranked list.
        ranking = order[order != q_idx]

        q_writer = writer_ids[q_idx]
        relevant_flags = writer_ids[ranking] == q_writer
        num_relevant = int(relevant_flags.sum())

        if expected_relevant is not None and num_relevant != expected_relevant:
            raise RuntimeError(
                f"Query {q_idx}: expected {expected_relevant} relevant items after self-removal, got {num_relevant}"
            )

        ap = average_precision_from_flags(relevant_flags)
        rr = reciprocal_rank_from_flags(relevant_flags)

        rel_positions = np.flatnonzero(relevant_flags)
        first_rank = int(rel_positions[0] + 1) if len(rel_positions) else -1

        aps.append(ap)
        rrs.append(rr)
        first_relevant_ranks.append(first_rank)

        row: Dict[str, Any] = {
            "query_index": int(q_idx),
            "query_page_id": str(page_ids[q_idx]),
            "query_writer_id": str(q_writer),
            "query_image_path": str(image_paths[q_idx]),
            "num_relevant": int(num_relevant),
            "ap": float(ap),
            "reciprocal_rank": float(rr),
            "first_relevant_rank": int(first_rank),
            "top1_index": int(ranking[0]),
            "top1_page_id": str(page_ids[ranking[0]]),
            "top1_writer_id": str(writer_ids[ranking[0]]),
            "top1_score": float(scores[ranking[0]]),
        }

        for k in top_ks:
            hit = bool(relevant_flags[:k].any())
            top_hits[k].append(hit)
            row[f"top{k}_hit"] = hit

        per_query_rows.append(row)

        if q_idx == 0 or (q_idx + 1) % 500 == 0:
            print(
                f"[EVAL] query={q_idx + 1}/{n} "
                f"AP={ap:.4f} RR={rr:.4f} first_rank={first_rank}"
            )

    aps_arr = np.asarray(aps, dtype=np.float64)
    rrs_arr = np.asarray(rrs, dtype=np.float64)
    first_ranks_arr = np.asarray(first_relevant_ranks, dtype=np.int64)

    metrics = {
        "num_queries": int(n),
        "descriptor_dim": int(dim),
        "num_writers": int(len(unique_writers)),
        "pages_per_writer_min": int(writer_counts.min()),
        "pages_per_writer_max": int(writer_counts.max()),
        "expected_relevant_per_query": int(expected_relevant) if expected_relevant is not None else None,
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

    csv_path = out_root / "per_query_results.csv"
    jsonl_path = out_root / "per_query_results.jsonl"

    print("[SAVE]")
    print("csv :", csv_path)
    print("jsonl:", jsonl_path)

    fieldnames = list(per_query_rows[0].keys())

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_query_rows:
            writer.writerow({k: to_py_scalar(v) for k, v in row.items()})

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in per_query_rows:
            f.write(json.dumps({k: to_py_scalar(v) for k, v in row.items()}, ensure_ascii=False) + "\n")

    summary = {
        "stage": "06R_C_retrieval_eval_resnet20_rsift_random500k_pca640",
        "methodological_status": "strict_rsift_random500k_test_only_leave_one_out_retrieval_pca640_cosine",
        "strict_branch": "strict_christlein2017_rsift_random500k",
        "descriptor_source": "Stage06R-B strict PCA640 whitened L2 descriptors",
        "retrieval_protocol": "ICDAR2017 Historical-WI TEST-only leave-one-out",
        "gallery": "all TEST pages except the query page itself",
        "relevant_items": "other TEST pages of the same writer",
        "self_match_excluded": True,
        "test_data_used_for_training_or_fitting": False,
        "descriptor_root": str(descriptor_root),
        "descriptor_path": str(test_desc_path),
        "writer_ids_path": str(test_writer_ids_path),
        "page_ids_path": str(test_page_ids_path),
        "output_root": str(out_root),
        "similarity": "dot_product_on_l2_normalized_descriptors_cosine_equivalent",
        "self_match_removed": True,
        "metrics": metrics,
        "per_query_csv": str(csv_path),
        "per_query_jsonl": str(jsonl_path),
        "validation": {
            "descriptors_finite": bool(np.isfinite(descriptors).all()),
            "descriptor_norm_mean": float(norms.mean()),
            "descriptor_norm_std": float(norms.std()),
            "all_ap_leq_1": bool(np.all(aps_arr <= 1.0 + 1e-12)),
            "all_num_relevant_expected": bool(
                expected_relevant is not None and all(int(r["num_relevant"]) == expected_relevant for r in per_query_rows)
            ),
        },
        "passed": True,
    }

    write_json(out_root / "retrieval_summary.json", summary)

    print("[METRICS]")
    print(json.dumps(metrics, indent=2))

    print("[VALIDATION]")
    print(json.dumps(summary["validation"], indent=2))

    print("[OK] Stage 06R-C strict-branch ResNet-20 retrieval evaluation completed.")
    print("[OUT]", out_root)


if __name__ == "__main__":
    main()
