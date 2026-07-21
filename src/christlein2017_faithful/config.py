from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FaithfulChristleinConfig:
    """Configuration constants for the faithful Christlein et al. (2017) reproduction."""

    project_root: Path = Path(".")
    output_root: Path = Path("outputs/faithful_christlein2017")

    train_images: int = 1182
    train_writers: int = 394
    test_images: int = 3600
    test_writers: int = 720

    patch_size: int = 32
    sift_pca_dim: int = 32

    surrogate_num_clusters: int = 5000
    surrogate_max_descriptors: int = 500_000
    surrogate_batch_size: int = 50_000
    surrogate_seed: int = 1
    ratio_threshold: float = 0.9
    ratio_sampling_seed: int = 42

    cnn_embedding_dim: int = 64

    mvlad_num_codebooks: int = 5
    rpca_regularization: float = 0.001

    esvm_c_grid: tuple[float, ...] = (
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    )


CFG = FaithfulChristleinConfig()
