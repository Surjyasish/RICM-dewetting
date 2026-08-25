"""
dw_ricm.py
==========
Dual-wavelength RICM height reconstruction for approximately axisymmetric
topographies (drops, caps, residual films).

Strategy
--------
1. Preprocess each channel independently (mask, envelope normalize).
2. Wrapped phase per channel via Larkin vortex transform with a
   structure-tensor orientation estimate.
3. Locate the apex from the smoothed fringe modulation.
4. 1D Itoh unwrap along many rays cast from the apex outward.
5. Median h(r) across rays -> axisymmetric height field h_axi(x, y).
6. Residual = wrap(psi_measured - psi_axi_predicted); unwrap with PUMA.
7. Total h = h_axi + h_residual.
8. Reference to contact-line plane, fix global sign with the dual-wavelength
   beat phase.

Public entry point:  reconstruct(I1, I2, lam_1, lam_2, n_med, px_size, ...)

Assumes the two intensity arrays are already isolated single-channel floats
of the same (H, W) shape and physically co-registered.
"""

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from scipy.fft import fft2, ifft2
from scipy.interpolate import PchipInterpolator

from puma2 import puma_unwrap


# ---------- preprocessing ----------

def build_mask(I, thr=20, erode=10):
    m = ndi.binary_opening(ndi.binary_fill_holes(I > thr), iterations=3)
    lbl, nl = ndi.label(m)
    sizes = ndi.sum(m, lbl, range(1, nl + 1))
    m = lbl == (np.argmax(sizes) + 1)
    return ndi.binary_erosion(m, iterations=erode)


def envelope_normalize(I, mask, sigma_bg=25, sigma_env=15):
    bg = ndi.gaussian_filter(I, sigma_bg)
    Ihp = (I - bg) * mask
    env = ndi.gaussian_filter(np.abs(Ihp), sigma_env) + 1e-6
    return (Ihp / env) * mask


# ---------- wrapped phase ----------

def _structure_tensor_beta(In, sigma=6):
    Ix = ndi.sobel(In, axis=1)
    Iy = ndi.sobel(In, axis=0)
    Jxx = ndi.gaussian_filter(Ix * Ix, sigma)
    Jyy = ndi.gaussian_filter(Iy * Iy, sigma)
    Jxy = ndi.gaussian_filter(Ix * Iy, sigma)
    return 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)


def vortex_wrapped_phase(In):
    H, W = In.shape
    beta = _structure_tensor_beta(In)
    u = np.fft.fftfreq(W)[None, :]
    v = np.fft.fftfreq(H)[:, None]
    mag = np.sqrt(u * u + v * v); mag[0, 0] = 1
    R = ifft2(((u + 1j * v) / mag) * fft2(In))
    q = -(np.cos(beta) * R.imag + np.sin(beta) * R.real)
    return np.arctan2(q, In)


# ---------- apex localization ----------

def locate_apex(In, mask, r_inner=40, r_outer=200,
                coarse_step=8, coh_thr=0.3):
    """
    Concentric-fringe center detector.

    At the apex, fringe normals β(x,y) should point radially:
    β mod π = atan2(y-c_y, x-c_x) mod π.
    Score(c) = mean of cos²(β - θ_c) over pixels in an annulus around c,
    weighted by orientation coherence. Argmax = apex.

    Robust across evaporation stages: dust bullseyes lose because the
    annulus radii sample much more of the drop's fringes than of a small
    defect's rings; late-stage residual caps still win over pinning noise.
    """
    H, W = In.shape
    sig = 8
    Ix = ndi.sobel(In, axis=1); Iy = ndi.sobel(In, axis=0)
    Jxx = ndi.gaussian_filter(Ix * Ix, sig)
    Jyy = ndi.gaussian_filter(Iy * Iy, sig)
    Jxy = ndi.gaussian_filter(Ix * Iy, sig)
    beta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    coh  = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-9)
    trust = mask & (coh > coh_thr)

    Y, X = np.mgrid[:H, :W]

    def score_at(cy, cx):
        dy = Y - cy; dx = X - cx
        r = np.hypot(dx, dy)
        sel = trust & (r > r_inner) & (r < r_outer)
        if sel.sum() < 300:
            return -np.inf
        theta = np.arctan2(dy[sel], dx[sel])
        return np.mean(np.cos(beta[sel] - theta) ** 2)

    best_s, best_yx = -np.inf, (H // 2, W // 2)
    for cy in range(coarse_step // 2, H, coarse_step):
        for cx in range(coarse_step // 2, W, coarse_step):
            s = score_at(cy, cx)
            if s > best_s: best_s, best_yx = s, (cy, cx)
    for dcy in range(-coarse_step, coarse_step + 1):
        for dcx in range(-coarse_step, coarse_step + 1):
            cy, cx = best_yx[0] + dcy, best_yx[1] + dcx
            if 0 <= cy < H and 0 <= cx < W:
                s = score_at(cy, cx)
                if s > best_s: best_s, best_yx = s, (cy, cx)
    return best_yx


# ---------- radial 1D unwrap ----------

def radial_unwrap(psi_wrapped, mask, apex_yx, n_rays=360, r_max=None,
                  step=0.5):
    """
    Cast n_rays from the apex outward, do 1D Itoh unwrap along each ray,
    return h(r) samples (one per ray at each radial bin).

    `step` is the ray sampling step in pixels.  Values below 1.0 protect
    against Nyquist-limited fringes near the drop edge where the wrap
    interval is close to one pixel.  0.5 is a safe default; drop to 0.33
    for very steep drops.
    """
    H, W = psi_wrapped.shape
    ay, ax = apex_yx
    if r_max is None:
        r_max = int(np.hypot(H, W))

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    # ray coordinate in pixels
    s = np.arange(0, r_max, step)
    # r_max output samples in units of 1 pixel (nearest whole-pixel bin)
    n_out = r_max
    r = np.arange(n_out, dtype=float)
    profiles = np.full((n_rays, n_out), np.nan)

    for i, th in enumerate(angles):
        xs = ax + s * np.cos(th)
        ys = ay + s * np.sin(th)
        # Bilinear interp of psi_w directly smooths across 2π jumps.
        # Interpolate exp(i*psi) as complex and take angle. Preserves wraps.
        cx = np.cos(psi_wrapped); sy = np.sin(psi_wrapped)
        c = ndi.map_coordinates(cx, [ys, xs], order=1, mode='constant', cval=np.nan)
        d = ndi.map_coordinates(sy, [ys, xs], order=1, mode='constant', cval=np.nan)
        vals = np.arctan2(d, c)
        msk = ndi.map_coordinates(mask.astype(float), [ys, xs], order=0,
                                  mode='constant', cval=0) > 0.5
        if msk.sum() < 5:
            continue
        first_gap = np.argmin(msk) if not msk.all() else msk.size
        vals = vals[:first_gap]
        unwrapped = np.unwrap(vals)
        s_kept = s[:first_gap]
        r_bins = np.floor(s_kept).astype(int)
        for rb in np.unique(r_bins):
            if rb >= n_out: break
            sel = (r_bins == rb)
            profiles[i, rb] = unwrapped[sel].mean()
    return r, profiles


def median_profile(r, profiles):
    """Median h(r) with IQR band."""
    med = np.nanmedian(profiles, axis=0)
    lo = np.nanpercentile(profiles, 25, axis=0)
    hi = np.nanpercentile(profiles, 75, axis=0)
    # trim to where at least a quarter of rays contribute
    n_valid = np.sum(~np.isnan(profiles), axis=0)
    good = n_valid >= max(10, profiles.shape[0] // 8)
    return r[good], med[good], lo[good], hi[good], n_valid[good]


# ---------- axisymmetric back-projection ----------

def build_axisymmetric_field(phi_med_r, r_grid, apex_yx, shape):
    """Interpolate the 1D radial profile back to a 2D field h_axi(x, y)."""
    H, W = shape
    Y, X = np.indices((H, W))
    r_pix = np.hypot(X - apex_yx[1], Y - apex_yx[0])
    if len(r_grid) < 4:
        return np.zeros(shape)
    # PCHIP is monotonicity-preserving; safer than cubic spline for cap profiles
    f = PchipInterpolator(r_grid, phi_med_r, extrapolate=False)
    phi_axi = f(r_pix)
    phi_axi[np.isnan(phi_axi)] = phi_med_r[-1]  # flat outside data
    return phi_axi


# ---------- sign resolution from beat ----------

def resolve_sign(psi1_w, psi2_w, phi_axi_ch1, lam_1, lam_2, mask, apex_yx):
    """
    Check whether phi_axi (from Ch1) has the correct sign by comparing its
    implied beat behavior with the measured beat phase near the apex.

    beat_measured = wrap(psi1 - psi2); with correct sign of h,
    beat should be nearly the same sign as phi_axi_ch1 * (1 - lam_1/lam_2)
    close to the apex.
    """
    # near-apex disk
    Y, X = np.indices(psi1_w.shape)
    d = np.hypot(X - apex_yx[1], Y - apex_yx[0])
    disk = mask & (d < 30)
    if disk.sum() < 20:
        return +1
    beat = np.angle(np.exp(1j * (psi1_w - psi2_w)))
    beat_med = np.median(beat[disk])
    axi_med  = np.median(phi_axi_ch1[disk]) * (1 - lam_1 / lam_2)
    return +1 if np.sign(beat_med) == np.sign(axi_med) or axi_med == 0 else -1


# ---------- entry point ----------

@dataclass
class DWRICMResult:
    h: np.ndarray            # final height field (m), contact-line-referenced
    h_axi: np.ndarray        # axisymmetric component (m)
    h_res: np.ndarray        # residual 2D component (m)
    r_grid: np.ndarray       # radial coordinate for profile (pixels)
    h_profile: np.ndarray    # median h(r) (m)
    h_profile_lo: np.ndarray
    h_profile_hi: np.ndarray
    apex_yx: tuple
    mask: np.ndarray
    contact_ring: np.ndarray
    contact_sigma_nm: float  # error floor from contact-line scatter
    global_sign: int


def reconstruct(I1, I2, lam_1, lam_2, n_med, px_size,
                mask_erode=10, apex_smooth=80, n_rays=360,
                apex_yx=None,
                do_residual=True, verbose=True):
    """
    apex_yx: optional (row, col) tuple to override auto-detection. Highly
             recommended for time series: detect on the first frame,
             then propagate.
    """
    I1 = I1.astype(float); I2 = I2.astype(float)

    # 1. Preprocess
    mask = build_mask(I1, erode=mask_erode) & build_mask(I2, erode=mask_erode)
    In1 = envelope_normalize(I1, mask)
    In2 = envelope_normalize(I2, mask)
    if verbose: print(f"mask: {mask.sum()} px")

    # 2. Wrapped phases
    psi1_w = vortex_wrapped_phase(In1) * mask
    psi2_w = vortex_wrapped_phase(In2) * mask

    # 3. Apex: user override or auto-detect
    if apex_yx is None:
        apex_yx = locate_apex(In1, mask)
    if verbose:
        print(f"apex at ({apex_yx[1]*px_size*1e3:.2f}, "
              f"{apex_yx[0]*px_size*1e3:.2f}) mm")

    # 4. Radial 1D unwrap on Ch1
    r, profiles = radial_unwrap(psi1_w, mask, apex_yx, n_rays=n_rays)
    r_grid, phi_med, phi_lo, phi_hi, n_valid = median_profile(r, profiles)
    # Convert phase to height (Ch1)
    h_profile = lam_1 * phi_med / (4 * np.pi * n_med)
    h_lo      = lam_1 * phi_lo  / (4 * np.pi * n_med)
    h_hi      = lam_1 * phi_hi  / (4 * np.pi * n_med)

    # 5. Axisymmetric back-projection
    phi_axi = build_axisymmetric_field(phi_med, r_grid, apex_yx, psi1_w.shape)
    h_axi   = lam_1 * phi_axi / (4 * np.pi * n_med) * mask

    # 6. Residual 2D unwrap: what does the *measured* wrapped phase say
    # after removing the axisymmetric prediction?
    if do_residual:
        psi_pred = np.angle(np.exp(1j * phi_axi))
        psi_res_w = np.angle(np.exp(1j * (psi1_w - psi_pred))) * mask
        phi_res = puma_unwrap(psi_res_w, mask=mask, T=3 * np.pi,
                              max_sweeps=8, verbose=False)
        h_res = lam_1 * phi_res / (4 * np.pi * n_med) * mask
    else:
        phi_res = np.zeros_like(psi1_w)
        h_res = np.zeros_like(h_axi)

    # 7. Total field before sign resolution
    phi_total = phi_axi + phi_res
    h_signed = lam_1 * phi_total / (4 * np.pi * n_med) * mask

    # 8. Contact-line plane reference (both sign candidates)
    mask_full = ndi.binary_fill_holes(I1 > 20) & ndi.binary_fill_holes(I2 > 20)
    dist_out = ndi.distance_transform_edt(mask_full)
    contact = mask & (dist_out <= 18) & (dist_out > 12)
    Yg, Xg = np.indices(h_signed.shape)
    A = np.column_stack([np.ones(contact.sum()),
                         Xg[contact].ravel(), Yg[contact].ravel()])

    def plane_reference(field):
        coef, *_ = np.linalg.lstsq(A, field[contact].ravel(), rcond=None)
        return (field - (coef[0] + coef[1] * Xg + coef[2] * Yg)) * mask

    h_pos = plane_reference(+h_signed)
    h_neg = plane_reference(-h_signed)

    # 9. Sign resolution: apex must be at least as high as contact line.
    # Compare a small disk around the apex against the contact ring median.
    disk = mask & (np.hypot(Xg - apex_yx[1], Yg - apex_yx[0]) < 20)
    def apex_score(h): return np.median(h[disk]) - np.median(h[contact])
    sign = +1 if apex_score(h_pos) >= apex_score(h_neg) else -1
    if verbose:
        print(f"apex score  +sign: {apex_score(h_pos)*1e6:+.2f} µm,  "
              f"-sign: {apex_score(h_neg)*1e6:+.2f} µm  ->  choose {sign:+d}")
    h_total = h_pos if sign == +1 else h_neg

    sigma_nm = np.std(h_total[contact]) * 1e9
    if verbose:
        print(f"h_apex = {h_total[apex_yx]*1e6:.2f} µm, "
              f"contact-ring σ = {sigma_nm:.0f} nm")

    return DWRICMResult(
        h=h_total, h_axi=h_axi, h_res=h_res,
        r_grid=r_grid, h_profile=h_profile,
        h_profile_lo=h_lo, h_profile_hi=h_hi,
        apex_yx=apex_yx, mask=mask, contact_ring=contact,
        contact_sigma_nm=float(sigma_nm), global_sign=int(sign))
