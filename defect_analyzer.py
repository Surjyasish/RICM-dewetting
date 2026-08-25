"""
defect_analyzer.py
==================
Detect and characterize dewetting defects (holes, trapped particles,
nucleated dry spots) in near-evaporation RICM frames.

Outputs per-defect morphometry and frame-level summary statistics:
- count, number density (per mm²)
- equivalent-diameter distribution
- area distribution
- nearest-neighbor spacing distribution and its ratio to a random
  (Poisson) expectation  -> tells ordered vs clustered vs random
- radial position distribution (are defects edge- or center-biased?)

Designed for time-series use: run on each frame, concatenate the
per-frame summaries to get dynamics (defect birth rate, coarsening).

A "defect" here = a compact dark feature against the fringe background.
Distinguishing a true dewetting hole from a trapped dust particle is not
possible from morphology alone; both are reported and can be filtered by
size or by darkness contrast downstream.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# ---------- masking ----------

def drop_mask(I, thr=20, erode=8):
    m = ndi.binary_opening(ndi.binary_fill_holes(I > thr), iterations=3)
    lbl, nl = ndi.label(m)
    sizes = ndi.sum(m, lbl, range(1, nl + 1))
    m = lbl == (np.argmax(sizes) + 1)
    return ndi.binary_erosion(m, iterations=erode)


# ---------- detection ----------

def detect_defects(I, mask, darkness_pct=8, min_area_px=4, max_area_px=400,
                   local_bg_sigma=15, contrast_frac=0.55,
                   min_circularity=0.4, mask_erode_extra=6):
    """
    Detect compact dark features.

    Stages:
    1. Global darkness threshold at `darkness_pct` percentile inside mask.
    2. Local-contrast confirmation against a median background.
    3. Extra mask erosion so the dark rim boundary is excluded.
    4. Circularity filter (4π·area / perimeter²) to reject elongated
       fragments (rim pieces, scratches); real dewetting holes are compact.

    Returns a contiguously-relabeled array of accepted defects.
    """
    inner = ndi.binary_erosion(mask, iterations=mask_erode_extra)
    Ivals = I[inner]
    thr = np.percentile(Ivals, darkness_pct)

    local_bg = ndi.median_filter(I, size=int(local_bg_sigma * 2 + 1))
    local_dark = I < contrast_frac * local_bg

    dark = inner & (I < thr) & local_dark
    dark = ndi.binary_opening(dark, iterations=1)

    lbl, n = ndi.label(dark)
    if n == 0:
        return np.zeros_like(lbl)

    idx = np.arange(1, n + 1)
    sizes = ndi.sum(dark, lbl, idx)

    # circularity via area and perimeter (perimeter from boundary pixel count,
    # corrected by a factor for discretization)
    keep = []
    for i in idx:
        a = sizes[i - 1]
        if a < min_area_px or a > max_area_px:
            continue
        region = (lbl == i)
        # perimeter: count boundary pixels of the region
        er = ndi.binary_erosion(region)
        perim = np.count_nonzero(region & ~er)
        if perim == 0:
            continue
        circ = 4 * np.pi * a / (perim ** 2)
        # discrete circularity of a disk is ~<1; use a lenient threshold
        if circ >= min_circularity:
            keep.append(i)

    clean = np.where(np.isin(lbl, keep), lbl, 0)
    clean, _ = ndi.label(clean > 0)
    return clean


# ---------- morphometry ----------

@dataclass
class DefectStats:
    n: int
    density_per_mm2: float
    centroids_mm: np.ndarray          # (n, 2) in mm, (x, y)
    equiv_diam_um: np.ndarray         # equivalent circular diameter
    area_um2: np.ndarray
    nn_dist_um: np.ndarray            # nearest-neighbor distance
    nn_ratio: float                   # observed mean NN / random expectation
    radial_pos_norm: np.ndarray       # 0 (center) .. 1 (edge)
    frame_area_mm2: float


def analyze(I, mask, px_size, **detect_kwargs):
    lbl = detect_defects(I, mask, **detect_kwargs)
    n = int(lbl.max())
    px_um = px_size * 1e6
    px_mm = px_size * 1e3
    frame_area_mm2 = mask.sum() * px_mm ** 2

    if n == 0:
        return DefectStats(0, 0.0, np.empty((0, 2)), np.array([]), np.array([]),
                           np.array([]), np.nan, np.array([]), frame_area_mm2), lbl

    # centroids and areas
    idx = np.arange(1, n + 1)
    centroids = np.array(ndi.center_of_mass(lbl > 0, lbl, idx))  # (row, col)
    areas_px = ndi.sum(lbl > 0, lbl, idx)

    area_um2 = areas_px * px_um ** 2
    equiv_diam_um = 2 * np.sqrt(area_um2 / np.pi)

    # centroids in mm as (x, y)
    cent_mm = np.column_stack([centroids[:, 1] * px_mm, centroids[:, 0] * px_mm])

    # nearest-neighbor distances
    if n >= 2:
        tree = cKDTree(cent_mm)
        d, _ = tree.query(cent_mm, k=2)
        nn = d[:, 1] * 1000.0  # mm -> µm
        # Poisson expectation for random points at same density:
        # E[NN] = 0.5 / sqrt(density)
        density_per_mm2 = n / frame_area_mm2
        e_nn_mm = 0.5 / np.sqrt(density_per_mm2)
        nn_ratio = np.mean(d[:, 1]) / e_nn_mm  # >1 ordered, <1 clustered
    else:
        nn = np.array([np.nan])
        nn_ratio = np.nan
        density_per_mm2 = n / frame_area_mm2

    # radial position (0 at drop centroid, 1 at max radius in mask)
    cy, cx = ndi.center_of_mass(mask)
    Y, X = np.indices(mask.shape)
    rmax = np.sqrt(((Y[mask] - cy) ** 2 + (X[mask] - cx) ** 2).max())
    r_def = np.sqrt((centroids[:, 0] - cy) ** 2 + (centroids[:, 1] - cx) ** 2)
    radial_norm = r_def / rmax

    stats = DefectStats(
        n=n, density_per_mm2=density_per_mm2, centroids_mm=cent_mm,
        equiv_diam_um=equiv_diam_um, area_um2=area_um2,
        nn_dist_um=nn, nn_ratio=nn_ratio, radial_pos_norm=radial_norm,
        frame_area_mm2=frame_area_mm2)
    return stats, lbl


# ---------- convenience: summarize one frame as a dict row ----------

def summary_row(stats, label=None):
    return dict(
        label=label,
        n_defects=stats.n,
        density_per_mm2=round(stats.density_per_mm2, 3),
        median_diam_um=round(float(np.median(stats.equiv_diam_um)), 2) if stats.n else np.nan,
        p90_diam_um=round(float(np.percentile(stats.equiv_diam_um, 90)), 2) if stats.n else np.nan,
        total_defect_area_mm2=round(float(stats.area_um2.sum() / 1e6), 4) if stats.n else 0.0,
        median_nn_um=round(float(np.nanmedian(stats.nn_dist_um)), 2) if stats.n >= 2 else np.nan,
        nn_ratio=round(float(stats.nn_ratio), 3) if stats.n >= 2 else np.nan,
        edge_fraction=round(float(np.mean(stats.radial_pos_norm > 0.66)), 3) if stats.n else np.nan,
    )
