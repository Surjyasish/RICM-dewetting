# ricm-dewetting

**Dual-wavelength RICM analysis of a residual liquid film near rupture.**

Dual-wavelength Reflection Interference Contrast Microscopy analysis of a residual liquid film shortly before complete evaporation. This repo contains the reconstruction pipeline, the analyzed frame, and reproducible outputs.

![raw Ch1 fringes](raw_ch1_t123.png)

## What this is

A single frame late in an evaporation series is neither a drop (topography reconstructable with the usual pipeline) nor a dry field (nothing to reconstruct). It is a **thin residual film perforated by dewetting defects**, with a coherent fringe region only over part of the wetted area. The analysis here:

1. Reconstructs **relative film thickness** in the coherent region, independently for each wavelength (488 nm and 561 nm).
2. Cross-validates the two wavelengths pixel-by-pixel and reports an internal error floor.
3. Detects and characterizes the **dewetting defects** as a separate morphometry problem.

Reconstructed thickness spans 0–3.9 µm with median ≈ 1.7 µm over 41 % of the wetted area. The two channels agree with RMS 0.29 µm, mean bias -0.01 µm, and pixel correlation r = 0.921.

## Why this frame is different

Classical DW-RICM height reconstruction (spiral-phase unwrap → PUMA → dual-wavelength beat) assumes a single coherent phase field over the whole wetted area. In this frame:

- Two-thirds of the drop is a near-dry stippled zone whose small-scale texture sits at or below the fringe coherence scale; no unwrap can extract meaningful topography there.
- The ~143 dewetting defects are true phase singularities and must be excluded from the unwrap, not smoothed over.
- The remaining coherent film is thin, so fringes are sparse and well above Nyquist — 2D unwrap **works quantitatively here**, unlike on the tall early-stage drops in the same series.

The pipeline therefore restricts the reconstruction to a coherent-and-defect-free region, and treats the defect field as its own quantitative product.

## Results at a glance

**Relative film thickness (dual-wavelength average, coherent region):**

![dualwave average](film_panels/09_dualwave_average.png)

**Cross-wavelength validation** (this is the honest error floor):

![Ch1 vs Ch2](film_panels/08_correlation.png)

The two wavelengths reconstruct the same physical film independently. RMS 0.29 µm ≈ 1.6 fringes (Ch1) or 1.4 fringes (Ch2). No systematic bias.

**Ch1 − Ch2 difference map:**

![difference](film_panels/06_difference.png)

The lower lobe (the trustworthy region) is uniformly pale. The systematic ~0.7 µm offset on the upper-left arc reflects that patch being disconnected from the lower lobe, so its absolute level is not fixed by the data — a limitation, not a bug.

## Repo layout

```
.
├── raw_ch1_t123.png                # raw 488 nm frame (visual reference)
├── film_thickness_report.md        # full technical report
├── film_thickness_dualwave.py      # CLI entry point (see below)
├── dw_ricm.py                      # mask / envelope / vortex / apex / radial unwrap
├── puma2.py                        # PUMA graph-cut phase unwrap
├── defect_analyzer.py              # dewetting-defect detection + morphometry
├── film_panels/                    # 9 stand-alone panel figures
│   ├── 01_raw_ch2.png
│   ├── 02_ch2_thickness.png
│   ├── 03_ch2_3d.png
│   ├── 04_ch1_common.png
│   ├── 05_ch2_common.png
│   ├── 06_difference.png
│   ├── 07_histograms.png
│   ├── 08_correlation.png
│   ├── 09_dualwave_average.png
│   └── film_thickness_dualwave.npz # all arrays + validation scalars
├── defects_t123.csv                # per-defect morphometry table
└── defect_analysis_v2.png          # 6-panel defect figure
```

## Reproducing

```bash
# dependencies
pip install numpy scipy matplotlib scikit-image tifffile PyMaxflow

# run
python film_thickness_dualwave.py CH1.tif CH2.tif --outdir results/
```

Optional arguments (defaults match this dataset):

| flag | default | meaning |
|---|---|---|
| `--lam1` | 488e-9 | Ch1 wavelength (m) |
| `--lam2` | 561e-9 | Ch2 wavelength (m) |
| `--n`    | 1.33   | medium refractive index |
| `--px`   | 9.983e-6 | pixel size (m) |
| `--ch1-plane` | 1 | RGB plane holding Ch1 signal (green = 1) |
| `--ch2-plane` | 0 | RGB plane holding Ch2 signal (red = 0) |

Outputs: nine standalone panel PNGs plus `film_thickness_dualwave.npz`.

## The `.npz` bundle

Load in Python:

```python
import numpy as np
d = np.load('film_panels/film_thickness_dualwave.npz')
h_avg  = d['h_avg']       # dual-λ averaged thickness, µm
common = d['common']      # valid pixel mask
print(f"RMS {float(d['rms']):.3f} µm | r {float(d['corr']):.3f}")
```

Keys:

| key | shape | meaning |
|---|---|---|
| `mask` | (H, W) bool | full wetted-drop mask |
| `h1`, `h2` | (H, W) float | per-channel relative thickness (µm), zero-referenced on each channel's own region |
| `reg1`, `reg2` | (H, W) bool | per-channel reconstructable region |
| `common` | (H, W) bool | `reg1 & reg2` |
| `h1c`, `h2c` | (H, W) float | thickness re-referenced on the common region |
| `h_avg` | (H, W) float | `0.5 * (h1c + h2c)` — recommended product |
| `diff` | (H, W) float | `h1c - h2c` |
| `coh1`, `coh2` | (H, W) float | fringe coherence maps |
| `rms`, `meandiff`, `corr` | scalar | validation scalars |
| `lam1`, `lam2`, `n_med`, `px_size` | scalar | calibration |

## Method summary

For each channel independently:

1. **Mask** the wetted region from intensity + morphology.
2. **Envelope-normalize** the fringes (Gaussian high-pass ÷ local envelope) to remove the illumination profile.
3. **Vortex (Larkin spiral-phase) transform** to extract wrapped phase from the single frame, using a structure-tensor orientation estimate for the quadrature.
4. **Restrict to the reconstructable region**: high orientation coherence AND not a dewetting defect. The upper two-thirds of the frame fails both tests and is excluded.
5. **PUMA graph-cut unwrap** on the reconstructable region. In this thin-film regime the fringes are well above Nyquist, so 2D PUMA is quantitatively reliable — unlike on the tall early-stage drops.
6. **Phase → thickness**: h = λ φ / (4 π n).
7. **Reference to zero minimum** (no negative thickness).

Cross-validation: re-reference both channels' thickness maps to a common zero over the common region, form the pixel-by-pixel difference and correlation, average for a noise-reduced product.

## Limitations

- **Relative thickness only.** Zero corresponds to the thinnest reconstructed pixel, not the dry substrate. Absolute thickness requires a dry-substrate reference in-frame.
- **Disconnected patches are not mutually referenced.** The two upper-rim arcs and the lower lobe are independently reconstructed; only within a patch is the thickness level absolute. The lower lobe is the largest and most reliable.
- **Sub-fringe features are uncertain in sign.** The dual-wavelength beat carries no useful coarse-height information when thickness variations are sub-fringe. The two wavelengths' agreement (r = 0.921) is the practical guardrail.
- **Defects vs particles.** The morphometry distinguishes size and position but not physical origin. Time-series persistence disambiguates them (particles stay put, holes grow).

## References

- Ghiglia, D. C., & Pritt, M. D. (1998). *Two-dimensional phase unwrapping: Theory, algorithms, and software*. Wiley.
- Bioucas-Dias, J. M., & Valadão, G. (2007). Phase unwrapping via graph cuts. *IEEE Trans. Image Process.*, 16(3), 698.
- Larkin, K. G., Bone, D. J., & Oldfield, M. A. (2001). Natural demodulation of two-dimensional fringe patterns. I. General background of the spiral phase quadrature transform. *J. Opt. Soc. Am. A*, 18(8), 1862.

## Note
Anybody using this repository please cite : Mitra, S., Kapur, V., Jones, L., & Mitra, S. K. (2025). Dynamics of an Artificial Tear Film on Contact Lenses in Response to a Moving Force Mimicking Fingertip Application. ACS Applied Materials & Interfaces, 17(44), 61426-61438.
