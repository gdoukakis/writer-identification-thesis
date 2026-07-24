from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


# Public/GitHub-safe default root.
# Expected location:
# phase2_christlein2017/src/christlein2017_faithful/00_prepare_manifests.py
# Therefore parents[2] resolves to phase2_christlein2017.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_MANIFESTS_DIR = PROJECT_ROOT / "manifests"
DATASETS_DIR = PROJECT_ROOT / "datasets"

OUTPUT_MANIFESTS_DIR = PROJECT_ROOT / "outputs" / "faithful_christlein2017" / "manifests"

TRAIN_MANIFEST_PATH = SOURCE_MANIFESTS_DIR / "manifest_train_1182.csv"
TEST_MANIFEST_PATH = SOURCE_MANIFESTS_DIR / "manifest_test_3600.csv"

TRAIN_IMAGES_DIR = DATASETS_DIR / "icdar17-historicalwi-training-binarized"
TEST_IMAGES_DIR = DATASETS_DIR / "ScriptNet-HistoricalWI-2017-binarized"

EXPECTED = {
    "train": {
        "rows": 1182,
        "writers": 394,
        "manifest_path": TRAIN_MANIFEST_PATH,
        "images_dir": TRAIN_IMAGES_DIR,
        "output_name": "train_manifest.csv",
    },
    "test": {
        "rows": 3600,
        "writers": 720,
        "manifest_path": TEST_MANIFEST_PATH,
        "images_dir": TEST_IMAGES_DIR,
        "output_name": "test_manifest.csv",
    },
}

REQUIRED_COLUMNS = ["path", "writer_id", "split"]

# If True, validate only manifest contents. This is useful for GitHub users
# because the Historical-WI images are not included in the repository.
VERIFY_MANIFESTS_ONLY = False


def parse_args() -> argparse.Namespace:
    """Parse public/GitHub-safe path configuration."""
    parser = argparse.ArgumentParser(
        description="Prepare validated ICDAR2017 Historical-WI manifests for Phase 2."
    )

    parser.add_argument(
        "--source-manifests-dir",
        type=Path,
        default=SOURCE_MANIFESTS_DIR,
        help=(
            "Directory containing manifest_train_1182.csv and "
            "manifest_test_3600.csv."
        ),
    )
    parser.add_argument(
        "--train-images-dir",
        type=Path,
        default=TRAIN_IMAGES_DIR,
        help="Directory containing the 1,182 binarised TRAIN pages.",
    )
    parser.add_argument(
        "--test-images-dir",
        type=Path,
        default=TEST_IMAGES_DIR,
        help="Directory containing the 3,600 binarised TEST pages.",
    )
    parser.add_argument(
        "--output-manifests-dir",
        type=Path,
        default=OUTPUT_MANIFESTS_DIR,
        help=(
            "Directory where train_manifest.csv, test_manifest.csv and "
            "dataset_summary.json will be written."
        ),
    )
    parser.add_argument(
        "--verify-manifests-only",
        action="store_true",
        help=(
            "Validate only the source manifest CSV files. "
            "Do not require local image directories or check image existence."
        ),
    )

    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    """Update the original module-level configuration from CLI arguments."""
    global SOURCE_MANIFESTS_DIR
    global OUTPUT_MANIFESTS_DIR
    global TRAIN_MANIFEST_PATH
    global TEST_MANIFEST_PATH
    global TRAIN_IMAGES_DIR
    global TEST_IMAGES_DIR
    global EXPECTED
    global VERIFY_MANIFESTS_ONLY

    SOURCE_MANIFESTS_DIR = args.source_manifests_dir
    OUTPUT_MANIFESTS_DIR = args.output_manifests_dir
    TRAIN_MANIFEST_PATH = SOURCE_MANIFESTS_DIR / "manifest_train_1182.csv"
    TEST_MANIFEST_PATH = SOURCE_MANIFESTS_DIR / "manifest_test_3600.csv"
    TRAIN_IMAGES_DIR = args.train_images_dir
    TEST_IMAGES_DIR = args.test_images_dir
    VERIFY_MANIFESTS_ONLY = bool(args.verify_manifests_only)

    EXPECTED = {
        "train": {
            "rows": 1182,
            "writers": 394,
            "manifest_path": TRAIN_MANIFEST_PATH,
            "images_dir": TRAIN_IMAGES_DIR,
            "output_name": "train_manifest.csv",
        },
        "test": {
            "rows": 3600,
            "writers": 720,
            "manifest_path": TEST_MANIFEST_PATH,
            "images_dir": TEST_IMAGES_DIR,
            "output_name": "test_manifest.csv",
        },
    }


def read_and_validate_manifest(split: str) -> pd.DataFrame:
    """Read, normalize, and validate one ICDAR17 manifest."""
    cfg = EXPECTED[split]
    manifest_path: Path = cfg["manifest_path"]
    images_dir: Path = cfg["images_dir"]

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not VERIFY_MANIFESTS_ONLY:
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

    df = pd.read_csv(manifest_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{manifest_path} is missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["path"] = df["path"].astype(str).str.strip()
    df["writer_id"] = pd.to_numeric(df["writer_id"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    invalid_split_rows = df[df["split"] != split]
    if len(invalid_split_rows) > 0:
        raise ValueError(
            f"Found {len(invalid_split_rows)} rows with wrong split in {manifest_path}. "
            f"Expected split='{split}'."
        )

    if len(df) != cfg["rows"]:
        raise ValueError(
            f"Unexpected number of rows for {split}: "
            f"expected {cfg['rows']}, got {len(df)}"
        )

    writer_count = df["writer_id"].nunique()
    if writer_count != cfg["writers"]:
        raise ValueError(
            f"Unexpected number of writers for {split}: "
            f"expected {cfg['writers']}, got {writer_count}"
        )

    df["filename"] = df["path"]
    df["image_id"] = df["filename"].apply(lambda x: Path(x).stem)

    if VERIFY_MANIFESTS_ONLY:
        df["image_path"] = df["filename"]
    else:
        df["image_path"] = df["filename"].apply(lambda x: str(images_dir / x))

        missing_files = df[~df["image_path"].apply(lambda x: Path(x).exists())]
        if len(missing_files) > 0:
            report_path = OUTPUT_MANIFESTS_DIR / f"missing_files_{split}.csv"
            OUTPUT_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
            missing_files.to_csv(report_path, index=False)

            examples = missing_files["image_path"].head(10).tolist()
            raise FileNotFoundError(
                f"Found {len(missing_files)} missing image files for split '{split}'. "
                f"Report written to: {report_path}. "
                f"First examples: {examples}"
            )

    df = df[["image_id", "filename", "writer_id", "split", "image_path"]].copy()
    return df


def describe_manifest(df: pd.DataFrame) -> Dict[str, object]:
    """Create a compact summary for one validated manifest."""
    per_writer = df.groupby("writer_id").size()
    extensions = df["filename"].apply(lambda x: Path(x).suffix.lower()).value_counts().to_dict()

    return {
        "rows": int(len(df)),
        "writers": int(df["writer_id"].nunique()),
        "images_per_writer_min": int(per_writer.min()),
        "images_per_writer_max": int(per_writer.max()),
        "images_per_writer_mean": float(per_writer.mean()),
        "extensions": {str(k): int(v) for k, v in extensions.items()},
    }


def write_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Write validated manifests and dataset summary."""
    OUTPUT_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    train_out = OUTPUT_MANIFESTS_DIR / EXPECTED["train"]["output_name"]
    test_out = OUTPUT_MANIFESTS_DIR / EXPECTED["test"]["output_name"]
    summary_out = OUTPUT_MANIFESTS_DIR / "dataset_summary.json"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    train_writers = set(train_df["writer_id"].tolist())
    test_writers = set(test_df["writer_id"].tolist())
    writer_overlap = sorted(train_writers.intersection(test_writers))

    summary = {
        "project_root": str(PROJECT_ROOT),
        "source_manifests": {
            "train": str(TRAIN_MANIFEST_PATH),
            "test": str(TEST_MANIFEST_PATH),
        },
        "image_directories": {
            "train": None if VERIFY_MANIFESTS_ONLY else str(TRAIN_IMAGES_DIR),
            "test": None if VERIFY_MANIFESTS_ONLY else str(TEST_IMAGES_DIR),
        },
        "validated_outputs": {
            "train": str(train_out),
            "test": str(test_out),
            "summary": str(summary_out),
        },
        "train": describe_manifest(train_df),
        "test": describe_manifest(test_df),
        "writer_overlap_count": len(writer_overlap),
        "writer_overlap_sample": writer_overlap[:20],
        "verify_manifests_only": bool(VERIFY_MANIFESTS_ONLY),
    }

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Validated manifests written successfully.")
    print(f"[OUT] {train_out}")
    print(f"[OUT] {test_out}")
    print(f"[OUT] {summary_out}")

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    """Prepare validated ICDAR17 manifests for the Christlein 2017 pipeline re-implementation."""
    args = parse_args()
    configure_from_args(args)

    train_df = read_and_validate_manifest("train")
    test_df = read_and_validate_manifest("test")
    write_outputs(train_df, test_df)


if __name__ == "__main__":
    main()
