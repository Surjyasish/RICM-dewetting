"""
film_thickness_dualwave.py
==========================
Dual-wavelength relative film-thickness reconstruction for near-evaporation
RICM frames, with cross-channel validation.

Pipeline per channel:
  1. Mask the wetted region.
  2. Envelope-normalize the fringes.
  3. Vortex (spiral-phase) wrapped phase.
  4. Restrict to the reconstructable region: high fringe coherence, not a
     dewetting defect. (Thin films have sparse, well-sampled fringes, so a
     direct 2D PUMA unwrap is quantitatively reliable here.)
  5. PUMA graph-cut unwrap on that region.
  6. Convert phase to relative thickness h = lambda * phi / (4 pi n).
  7. Reference so the minimum is zero (no negative thickness).

Cross-validation:
  The two wavelengths reconstruct the same physical film independently.
  Re-reference both on the common region, compare pixel-by-pixel, and form
  the noise-reduced average.

Depends on: dw_ricm.py, puma2.py, defect_analyzer.py (same directory).

Usage:
    python film_thickness_dualwave.py CH1.tif CH2.tif --outdir results/
"""

import argparse
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from dw_ricm import build_mask, envelope_normalize, vortex_wrapped_phase
from puma2 import puma_unwrap
from defect_analyzer import detect_defects


# ----------------------------- core -----------------------------

def coherence_map(In, mask, sigma=8, smooth=12):
    """Structure-tensor orientation coherence, smoothed. High where the
    fringes are locally well-defined; low in stippled / near-dry regions."""
    Ix = ndi.sobel(In, axis=1); Iy = ndi.sobel(In, axis=0)
    Jxx = ndi.gaussian_filter(Ix * Ix, sigma)
    Jyy = ndi.gaussian_filter(Iy * Iy, sigma)
    Jxy = ndi.gaussian_filter(Ix * Iy, sigma)
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-9)
    return ndi.gaussian_filter(coh * mask, smooth)


def reconstruct_channel(I, mask, lam, n_med,
                        coh_thr=0.4, min_region_px=2000,
                        defect_dilate=2, puma_T=3 * np.pi, puma_sweeps=12):
    """Return relative thickness (µm), the reconstructable region mask, and
    the coherence map, for one wavelength channel."""
    In = envelope_normalize(I, mask)
    psi = vortex_wrapped_phase(In) * mask

    coh_s = coherence_map(In, mask)
    defects = ndi.binary_dilation(detect_defects(I, mask) > 0,
                                  iterations=defect_dilate)
    region = mask & (coh_s > coh_thr) & (~defects)
    region = ndi.binary_closing(ndi.binary_opening(region, iterations=2),
                                iterations=4)
    lbl, nl = ndi.label(region)
    if nl > 0:
        sizes = ndi.sum(region, lbl, range(1, nl + 1))
        region = np.isin(lbl, 1 + np.where(sizes > min_region_px)[0])

    phi = puma_unwrap(psi, mask=region, T=puma_T,
                      max_sweeps=puma_sweeps, verbose=False)
    h = lam * phi / (4 * np.pi * n_med)
    if region.sum() > 0:
        h = h - h[region].min()          # relative thickness, min = 0
    return h * 1e6, region, coh_s


def dual_wavelength(I1, I2, lam1, lam2, n_med, px_size, erode=10):
    """Full dual-channel reconstruction. Returns a dict of arrays and scalars."""
    mask = build_mask(I1, erode=erode) & build_mask(I2, erode=erode)
    h1, reg1, coh1 = reconstruct_channel(I1, mask, lam1, n_med)
    h2, reg2, coh2 = reconstruct_channel(I2, mask, lam2, n_med)

    common = reg1 & reg2
    # re-reference both to zero-min on the COMMON region for fair comparison
    h1c = h1 - h1[common].min() if common.sum() else h1
    h2c = h2 - h2[common].min() if common.sum() else h2
    diff = h1c - h2c
    h_avg = 0.5 * (h1c + h2c)

    if common.sum() >= 2:
        rms = float(np.sqrt(np.mean(diff[common] ** 2)))
        meandiff = float(np.mean(diff[common]))
        r = float(np.corrcoef(h1c[common], h2c[common])[0, 1])
    else:
        rms = meandiff = r = np.nan

    return dict(mask=mask, h1=h1, reg1=reg1, coh1=coh1,
                h2=h2, reg2=reg2, coh2=coh2,
                common=common, h1c=h1c, h2c=h2c, diff=diff, h_avg=h_avg,
                rms=rms, meandiff=meandiff, corr=r,
                lam1=lam1, lam2=lam2, n_med=n_med, px_size=px_size)


# ----------------------------- plotting -----------------------------

def _extent(shape, px):
    return np.array([0, shape[1], shape[0], 0]) * px * 1e3


def save_individual_plots(R, outdir, load_ch1_raw, load_ch2_raw):
    """Render each analysis view as a standalone figure."""
    os.makedirs(outdir, exist_ok=True)
    px = R['px_size']; n = R['n_med']
    ext = _extent(R['mask'].shape, px)
    reg1, reg2, common = R['reg1'], R['reg2'], R['common']

    def masked(a, m): return np.where(m, a, np.nan)

    def one(fname, fn):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fn(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=130, bbox_inches='tight')
        plt.close(fig)

    # 1. raw Ch2
    def p1(ax):
        ax.imshow(load_ch2_raw, cmap='gray', extent=ext)
        ax.set_title('Raw Ch2 (561 nm)'); ax.set_xlabel('mm'); ax.set_ylabel('mm')
    one('01_raw_ch2.png', p1)

    # 2. Ch2 relative thickness
    def p2(ax):
        v = np.nanpercentile(masked(R['h2'], reg2), 99)
        im = ax.imshow(masked(R['h2'], reg2), cmap='viridis', vmin=0, vmax=v, extent=ext)
        ax.set_title(f'Ch2 relative thickness (Δ={R["lam2"]/(2*n)*1e6:.3f} µm/fringe)')
        ax.set_xlabel('mm'); ax.set_ylabel('mm')
        plt.colorbar(im, ax=ax, fraction=0.045, label='µm')
    one('02_ch2_thickness.png', p2)

    # 3. Ch2 3D
    def p3(ax):
        pass  # handled separately (needs 3d projection)
    fig = plt.figure(figsize=(6.5, 5.5)); ax = fig.add_subplot(111, projection='3d')
    step = 5
    Y3, X3 = np.mgrid[:R['mask'].shape[0]:step, :R['mask'].shape[1]:step]
    h3 = masked(R['h2'], reg2)[::step, ::step]
    v = np.nanpercentile(masked(R['h2'], reg2), 99)
    ax.plot_surface(X3*px*1e3, Y3*px*1e3, h3, cmap='viridis', vmin=0, vmax=v, edgecolor='none')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('h (µm)')
    ax.set_title('Ch2 3D relative thickness'); ax.view_init(elev=40, azim=-70)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, '03_ch2_3d.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    # 4. Ch1 on common
    def p4(ax):
        v = np.nanpercentile(R['h1c'][common], 99)
        im = ax.imshow(masked(R['h1c'], common), cmap='viridis', vmin=0, vmax=v, extent=ext)
        ax.set_title('Ch1 (488 nm) on common region'); ax.set_xlabel('mm'); ax.set_ylabel('mm')
        plt.colorbar(im, ax=ax, fraction=0.045, label='µm')
    one('04_ch1_common.png', p4)

    # 5. Ch2 on common
    def p5(ax):
        v = np.nanpercentile(R['h2c'][common], 99)
        im = ax.imshow(masked(R['h2c'], common), cmap='viridis', vmin=0, vmax=v, extent=ext)
        ax.set_title('Ch2 (561 nm) on common region'); ax.set_xlabel('mm'); ax.set_ylabel('mm')
        plt.colorbar(im, ax=ax, fraction=0.045, label='µm')
    one('05_ch2_common.png', p5)

    # 6. difference
    def p6(ax):
        lim = np.nanpercentile(np.abs(R['diff'][common]), 98)
        im = ax.imshow(masked(R['diff'], common), cmap='RdBu_r', vmin=-lim, vmax=lim, extent=ext)
        ax.set_title(f'Ch1 − Ch2 difference (RMS {R["rms"]:.3f} µm)')
        ax.set_xlabel('mm'); ax.set_ylabel('mm')
        plt.colorbar(im, ax=ax, fraction=0.045, label='µm')
    one('06_difference.png', p6)

    # 7. histograms
    def p7(ax):
        ax.hist(R['h1'][reg1], bins=60, color='steelblue', alpha=0.6, label='Ch1 488 nm')
        ax.hist(R['h2'][reg2], bins=60, color='indianred', alpha=0.6, label='Ch2 561 nm')
        ax.set_xlabel('relative thickness (µm)'); ax.set_ylabel('pixel count')
        ax.set_title('Thickness distributions'); ax.legend()
    one('07_histograms.png', p7)

    # 8. correlation scatter
    def p8(ax):
        idx = np.where(common.ravel())[0]
        sub = np.random.choice(idx, size=min(8000, idx.size), replace=False)
        ax.scatter(R['h1c'].ravel()[sub], R['h2c'].ravel()[sub], s=2, alpha=0.2)
        mx = max(np.nanpercentile(R['h1c'][common], 99), np.nanpercentile(R['h2c'][common], 99))
        ax.plot([0, mx], [0, mx], 'r--', lw=1, label='1:1')
        ax.set_xlabel('Ch1 thickness (µm)'); ax.set_ylabel('Ch2 thickness (µm)')
        ax.set_title(f'Ch1 vs Ch2 (r = {R["corr"]:.3f})'); ax.legend(); ax.set_aspect('equal')
    one('08_correlation.png', p8)

    # 9. dual-wavelength averaged
    def p9(ax):
        v = np.nanpercentile(R['h_avg'][common], 99)
        im = ax.imshow(masked(R['h_avg'], common), cmap='viridis', vmin=0, vmax=v, extent=ext)
        ax.set_title('Dual-λ averaged thickness (noise-reduced)')
        ax.set_xlabel('mm'); ax.set_ylabel('mm')
        plt.colorbar(im, ax=ax, fraction=0.045, label='µm')
    one('09_dualwave_average.png', p9)


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ch1'); ap.add_argument('ch2')
    ap.add_argument('--outdir', default='film_results')
    ap.add_argument('--lam1', type=float, default=488e-9)
    ap.add_argument('--lam2', type=float, default=561e-9)
    ap.add_argument('--n', type=float, default=1.33)
    ap.add_argument('--px', type=float, default=9.983e-6)
    ap.add_argument('--ch1-plane', type=int, default=1,
                    help='RGB plane holding the Ch1 signal (green=1)')
    ap.add_argument('--ch2-plane', type=int, default=0,
                    help='RGB plane holding the Ch2 signal (red=0)')
    args = ap.parse_args()

    raw1 = tifffile.imread(args.ch1)
    raw2 = tifffile.imread(args.ch2)
    I1 = (raw1[..., args.ch1_plane] if raw1.ndim == 3 else raw1).astype(float)
    I2 = (raw2[..., args.ch2_plane] if raw2.ndim == 3 else raw2).astype(float)

    R = dual_wavelength(I1, I2, args.lam1, args.lam2, args.n, args.px)

    print(f"Ch1 region {R['reg1'].sum()} px, median {np.median(R['h1'][R['reg1']]):.2f} µm")
    print(f"Ch2 region {R['reg2'].sum()} px, median {np.median(R['h2'][R['reg2']]):.2f} µm")
    print(f"common {R['common'].sum()} px | RMS {R['rms']:.3f} µm | "
          f"mean diff {R['meandiff']:+.3f} µm | r {R['corr']:.3f}")

    save_individual_plots(R, args.outdir, I1, I2)
    np.savez(os.path.join(args.outdir, 'film_thickness_dualwave.npz'),
             **{k: v for k, v in R.items() if isinstance(v, np.ndarray)},
             rms=R['rms'], meandiff=R['meandiff'], corr=R['corr'],
             lam1=R['lam1'], lam2=R['lam2'], n_med=R['n_med'], px_size=R['px_size'])
    print(f"wrote plots + npz to {args.outdir}/")


if __name__ == '__main__':
    main()
