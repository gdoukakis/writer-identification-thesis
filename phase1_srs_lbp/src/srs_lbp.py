import numpy as np
from typing import Iterable, List, Literal, Tuple, Optional


DiscardZeroMode = Literal["keep_256_zeroed", "drop_255"]


def _bilinear_interpolate(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation for a single-channel image.

    Parameters
    ----------
    image : np.ndarray
        2D array (H, W), float32/float64 recommended.
    y, x : np.ndarray
        Sample coordinates (same shape), float arrays.

    Returns
    -------
    np.ndarray
        Interpolated values at (y, x), same shape as y/x.
    """
    h, w = image.shape

    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def _otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """
    Otsu threshold for a 1D array of non-negative values.

    Notes
    -----
    This implementation operates on a histogram with `nbins` bins spanning [min, max].
    """
    v = values.astype(np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0

    vmin = float(v.min())
    vmax = float(v.max())
    if vmax <= vmin + 1e-12:
        return vmin

    hist, bin_edges = np.histogram(v, bins=nbins, range=(vmin, vmax), density=False)
    hist = hist.astype(np.float64)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Avoid division by zero
    nonzero1 = weight1 > 0
    nonzero2 = weight2 > 0

    mean1 = np.zeros_like(bin_centers)
    mean2 = np.zeros_like(bin_centers)

    mean1[nonzero1] = np.cumsum(hist * bin_centers)[nonzero1] / weight1[nonzero1]
    mean2[nonzero2] = (np.cumsum((hist * bin_centers)[::-1])[::-1][nonzero2] / weight2[nonzero2])

    # Between-class variance
    sigma_b2 = weight1 * weight2 * (mean1 - mean2) ** 2

    # Maximize variance
    idx = int(np.argmax(sigma_b2))
    return float(bin_centers[idx])


def _srs_lbp_codes_for_radius(
    image: np.ndarray,
    radius: int,
    points: int = 8,
    use_otsu_threshold: bool = True,
    fixed_threshold: Optional[float] = None,
    valid_border: bool = True,
) -> np.ndarray:
    """
    Compute SRS-LBP codes for a single radius.

    Paper-faithful choices:
    - points should be 8 (SRS keeps P constant)
    - threshold t is computed via Otsu on |gc - gp| statistics (per image, per radius)

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (H, W), float32 recommended.
    radius : int
        Radius R.
    points : int
        Number of sampling points P on the circle.
    use_otsu_threshold : bool
        If True, compute t_hat using Otsu over |gc-gp|.
    fixed_threshold : Optional[float]
        If provided and use_otsu_threshold is False, use this threshold.
    valid_border : bool
        If True, compute codes only for pixels where all sampled points are inside the image
        (i.e., ignore border). If False, allow clamped sampling via interpolation boundaries.

    Returns
    -------
    np.ndarray
        2D array of codes (H', W') depending on border handling, dtype uint8.
    """
    if image.ndim != 2:
        raise ValueError("image must be a 2D array (grayscale).")

    img = image.astype(np.float32, copy=False)
    h, w = img.shape

    if valid_border:
        y0, y1 = radius, h - radius
        x0, x1 = radius, w - radius
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"Image too small for radius={radius}.")
        gc = img[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    else:
        gc = img
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    angles = 2.0 * np.pi * np.arange(points, dtype=np.float32) / float(points)
    # Standard LBP ordering: p=0..P-1
    dy = -radius * np.sin(angles)
    dx =  radius * np.cos(angles)

    diffs_abs_list = []
    gp_list = []

    for p in range(points):
        y = yy + dy[p]
        x = xx + dx[p]
        gp = _bilinear_interpolate(img, y, x)
        gp_list.append(gp)
        diffs_abs_list.append(np.abs(gc - gp))

    diffs_abs = np.stack(diffs_abs_list, axis=0)  # (P, H', W')
    if use_otsu_threshold:
        t_hat = _otsu_threshold(diffs_abs.reshape(-1))
    else:
        t_hat = 0.0 if fixed_threshold is None else float(fixed_threshold)

    bits = (diffs_abs >= t_hat).astype(np.uint8)  # (P, H', W')
    # Convert bits to codes
    weights = (1 << np.arange(points, dtype=np.uint8))[:, None, None]
    codes = np.sum(bits * weights, axis=0).astype(np.uint8)  # (H', W')

    return codes


def srs_lbp_histogram_embedding(
    image: np.ndarray,
    radii: Iterable[int] = range(1, 13),
    points: int = 8,
    discard_zero_mode: DiscardZeroMode = "keep_256_zeroed",
    l1_normalize_blocks: bool = True,
    valid_border: bool = True,
) -> np.ndarray:
    """
    Compute the concatenated SRS-LBP histogram embedding over multiple radii.

    The paper states:
    - histogram per radius
    - discard the zero pattern
    - L1 normalize and concatenate all histograms
    - descriptor size reported as 256 * |R|

    Because "discard zero" conflicts with "256 * |R|", we support two modes:
    - keep_256_zeroed: keep 256 bins but set bin[0]=0 before normalization
    - drop_255: drop bin[0] and keep 255 bins per radius

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (H, W).
    radii : Iterable[int]
        Radii to use (e.g., 1..12).
    points : int
        Number of sampling points (must be 8 for SRS-LBP as defined).
    discard_zero_mode : {"keep_256_zeroed", "drop_255"}
        How to handle the "discard zero pattern" statement.
    l1_normalize_blocks : bool
        If True, apply L1 normalization to each radius histogram block.
    valid_border : bool
        If True, compute codes only where circle fits inside the image.

    Returns
    -------
    np.ndarray
        1D embedding vector, float32.
        Length is:
        - 256 * len(radii) for keep_256_zeroed
        - 255 * len(radii) for drop_255
    """
    if points != 8:
        raise ValueError("For SRS-LBP, points must be 8 (paper definition keeps P=8 constant).")

    blocks: List[np.ndarray] = []
    for r in radii:
        codes = _srs_lbp_codes_for_radius(
            image=image,
            radius=int(r),
            points=points,
            use_otsu_threshold=True,
            fixed_threshold=None,
            valid_border=valid_border,
        )

        # Histogram over all codes (global pooling)
        hist = np.bincount(codes.reshape(-1), minlength=256).astype(np.float32)

        if discard_zero_mode == "keep_256_zeroed":
            hist[0] = 0.0
            if l1_normalize_blocks:
                s = hist.sum()
                if s > 0:
                    hist /= s
            blocks.append(hist)

        elif discard_zero_mode == "drop_255":
            hist255 = hist[1:].copy()
            if l1_normalize_blocks:
                s = hist255.sum()
                if s > 0:
                    hist255 /= s
            blocks.append(hist255)

        else:
            raise ValueError(f"Unknown discard_zero_mode: {discard_zero_mode}")

    emb = np.concatenate(blocks, axis=0).astype(np.float32)
    return emb
