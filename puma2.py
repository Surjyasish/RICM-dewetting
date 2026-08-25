"""
PUMA phase unwrap  --  correct alpha-expansion.

For alpha in {+1, -1}, propose x_i in {0,1} where x_i=1 means k_i += alpha.
Per-clique energies use truncated quadratic V(t) = min(t^2, T^2) on the
unwrapped phase differences  phi_i - phi_j = (psi_i + 2 pi k_i) - (psi_j + 2 pi k_j).

For each pair (i, j) with energies e_00, e_01, e_10, e_11:
  Constant  = e_00  (absorbed)
  Unary_i   = (e_10 - e_00) * x_i
  Unary_j   = (e_11 - e_10) * x_j
  Pairwise  = (e_01 + e_10 - e_00 - e_11) * (1 - x_i) * x_j    (needs >= 0)
When the pairwise coefficient is negative (non-submodular), the graph-cut
minimum is a lower bound; we truncate at 0 which is standard.
"""
import numpy as np
import maxflow


def _V(t, T):
    return np.minimum(t * t, T * T)


def puma_unwrap(psi_wrapped, mask=None, T=3 * np.pi, max_sweeps=15, verbose=True):
    psi = np.asarray(psi_wrapped, float)
    H, W = psi.shape
    if mask is None:
        mask = np.ones((H, W), bool)
    k = np.zeros((H, W), np.int32)

    def energy(k_):
        s = psi + 2 * np.pi * k_
        dh = s[:, :-1] - s[:, 1:]
        dv = s[:-1, :] - s[1:, :]
        mh = mask[:, :-1] & mask[:, 1:]
        mv = mask[:-1, :] & mask[1:, :]
        return _V(dh[mh], T).sum() + _V(dv[mv], T).sum()

    E = energy(k)
    if verbose:
        print(f"  initial E = {E:.4g}")

    # Node indices for in-mask pixels
    idx = -np.ones((H, W), np.int64)
    in_pix = np.flatnonzero(mask.ravel())
    idx.ravel()[in_pix] = np.arange(in_pix.size)
    N = in_pix.size

    for sweep in range(max_sweeps):
        improved = False
        for alpha in (+1, -1):
            s0 = psi + 2 * np.pi * k
            s1 = s0 + 2 * np.pi * alpha  # phase if that pixel takes the move

            g = maxflow.Graph[float]()
            g.add_nodes(N)
            # Accumulators: net source-to-node and node-to-sink capacity
            src = np.zeros(N)   # source -> node capacity  (paid if node ends on T side, x=1)
            snk = np.zeros(N)   # node -> sink capacity    (paid if node ends on S side, x=0)

            # Build pair contributions for both directions
            for direction in ('h', 'v'):
                if direction == 'h':
                    A = idx[:, :-1]; B = idx[:, 1:]
                    d00 = s0[:, :-1] - s0[:, 1:]
                    d01 = s0[:, :-1] - s1[:, 1:]
                    d10 = s1[:, :-1] - s0[:, 1:]
                    d11 = s1[:, :-1] - s1[:, 1:]
                else:
                    A = idx[:-1, :]; B = idx[1:, :]
                    d00 = s0[:-1, :] - s0[1:, :]
                    d01 = s0[:-1, :] - s1[1:, :]
                    d10 = s1[:-1, :] - s0[1:, :]
                    d11 = s1[:-1, :] - s1[1:, :]
                v = (A >= 0) & (B >= 0)
                Ai = A[v]; Bi = B[v]
                e00 = _V(d00[v], T)
                e01 = _V(d01[v], T)
                e10 = _V(d10[v], T)
                e11 = _V(d11[v], T)

                # Unary on i:  (e10 - e00) * x_i
                u_i = e10 - e00
                np.add.at(src, Ai, np.where(u_i > 0, u_i, 0))
                np.add.at(snk, Ai, np.where(u_i < 0, -u_i, 0))
                # Unary on j:  (e11 - e10) * x_j
                u_j = e11 - e10
                np.add.at(src, Bi, np.where(u_j > 0, u_j, 0))
                np.add.at(snk, Bi, np.where(u_j < 0, -u_j, 0))
                # Pairwise edge (i -> j) with cap max(0, e01+e10-e00-e11)
                w_ij = np.clip(e01 + e10 - e00 - e11, 0.0, None)
                # Vectorized edge insertion isn't supported directly; loop only over nonzero edges
                nz = w_ij > 0
                for a_, b_, w_ in zip(Ai[nz], Bi[nz], w_ij[nz]):
                    g.add_edge(int(a_), int(b_), float(w_), 0.0)

            # Apply accumulated tedges
            for i, (cs, ct) in enumerate(zip(src, snk)):
                if cs > 0 or ct > 0:
                    g.add_tedge(int(i), float(cs), float(ct))

            g.maxflow()
            seg = g.get_grid_segments(np.arange(N))  # False = source (x=0), True = sink (x=1)
            move_full = np.zeros(H * W, bool)
            move_full[in_pix] = seg
            k_new = k.copy()
            k_new.ravel()[move_full] += alpha
            E_new = energy(k_new)
            if E_new + 1e-9 < E:
                if verbose:
                    print(f"  sweep {sweep:2d} alpha={alpha:+d}: E {E:.4g} -> {E_new:.4g}  ({int(seg.sum())} px)")
                k = k_new; E = E_new; improved = True
        if not improved:
            if verbose:
                print(f"  converged after sweep {sweep}")
            break

    return (psi + 2 * np.pi * k) * mask
