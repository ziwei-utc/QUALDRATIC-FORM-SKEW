#!/usr/bin/env python3
"""Reproduce Figure 2 and Table 2 of Ma and Wang, "Quadratic Forms
under Skew-Normal and Skew-Elliptical Models: A Unified Survey".

U = X'X with X ~ SN_k(nu, I_k, alpha) is simulated from the
additive representation X = nu + delta*|T| + V, where
delta = alpha / sqrt(1 + alpha'alpha), T ~ N(0, 1) and
V ~ N_k(0, I_k - delta delta') are independent.

Output: Table 2 (closed forms (13)-(14) next to Monte Carlo
estimates), the central-case density gap (Remark 1), and Figure 2.
Usage:  python skewchi_fig2_table2.py [--n N] [--seed S] [--out F]
Tested: Python 3.12.7, NumPy 1.26.4, Matplotlib 3.9.2 (about 4 s).
"""
import argparse
import math

import numpy as np

K = 4                    # degrees of freedom
LAM = 3.0                # noncentrality, left panel
SHAPES = (0.0, 1.0, 3.0) # alpha = a 1_k, delta2 = 0, 4, 36
GRID_MAX, N_BINS, SMOOTH = 30.0, 600, 9


def simulate_skewchi(rng, nu, alpha, n):
    """n draws of U = X'X with X ~ SN_k(nu, I_k, alpha)."""
    k = nu.size
    delta = alpha / math.sqrt(1.0 + alpha @ alpha)
    chol = np.linalg.cholesky(np.eye(k) - np.outer(delta, delta))
    t_abs = np.abs(rng.standard_normal(n))
    v = rng.standard_normal((n, k)) @ chol.T
    x = nu[None, :] + np.outer(t_abs, delta) + v
    return np.einsum("ij,ij->i", x, x)


def theory_moments(lam, k, an, aa):
    """Mean and variance (13)-(14).

    an = alpha'nu (delta1), aa = alpha'alpha (delta2).
    """
    s = math.sqrt(1.0 + aa)
    mean = lam + k + 4.0 * an / (math.sqrt(2.0 * math.pi) * s)
    corr = (4.0 * an * (math.sqrt(2.0 * math.pi) * (aa + 2.0)
                        - 2.0 * s * an)
            / (math.pi * s ** 3))
    return mean, 4.0 * lam + 2.0 * k + corr


def mc_moments(u):
    """Sample mean and variance of u, with standard errors."""
    n, mean, var = u.size, u.mean(), u.var()
    m4 = np.mean((u - mean) ** 4)
    se_mean = u.std() / math.sqrt(n)
    se_var = math.sqrt((m4 - var ** 2) / n)
    return mean, se_mean, var, se_var


def density(u):
    """Histogram density on [0, GRID_MAX], raw and smoothed."""
    h, edges = np.histogram(u, bins=N_BINS, range=(0.0, GRID_MAX),
                            density=True)
    smooth = np.convolve(h, np.ones(SMOOTH) / SMOOTH, mode="same")
    return 0.5 * (edges[:-1] + edges[1:]), h, smooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--out", default="fig_skewchi.pdf")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    nu_nc = np.full(K, math.sqrt(LAM / K))   # lambda = nu'nu = 3
    nu_c = np.zeros(K)                       # central case
    runs = [(nu_nc, a) for a in SHAPES] + [(nu_c, SHAPES[-1])]

    # Table 2
    print(f"Table 2: k={K}, N={args.n}, seed={args.seed}")
    print("lam d2     d1  E(th)  E(MC) (se)   V(th)  V(MC) (se)")
    samples = {}
    for nu, a in runs:
        alpha = np.full(K, a)
        lam, an = float(nu @ nu), float(alpha @ nu)
        aa = float(alpha @ alpha)
        u = simulate_skewchi(rng, nu, alpha, args.n)
        samples[(round(lam), round(aa))] = u
        e_th, v_th = theory_moments(lam, K, an, aa)
        m, se_m, v, se_v = mc_moments(u)
        print(f"{lam:3.0f} {aa:2.0f} {an:6.3f} {e_th:6.3f} "
              f"{m:6.3f} ({se_m:.3f}) {v_th:7.3f} "
              f"{v:7.3f} ({se_v:.3f})")

    # Figure 2 curves and the Remark 1 check
    curves = {}
    for lam, nu in ((LAM, nu_nc), (0.0, nu_c)):
        for a in SHAPES:
            key = (round(lam), round(K * a * a))
            u = samples.get(key)
            if u is None:
                u = simulate_skewchi(rng, nu, np.full(K, a),
                                     args.n)
            curves[key] = density(u)
    d0, d36 = curves[(0, 0)], curves[(0, 36)]
    print("central case, max |density gap|, delta2 = 0 vs 36: "
          f"raw {np.max(np.abs(d0[1] - d36[1])):.4f}, "
          f"smoothed {np.max(np.abs(d0[2] - d36[2])):.4f}")
    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    titles = (rf"Noncentral case: $\lambda$ = {LAM:g}, k = {K}",
              rf"Central case: $\lambda$ = 0, k = {K}"
              " (curves coincide)")
    for ax, lam, title in zip(axes, (LAM, 0.0), titles):
        for a in SHAPES:
            d2 = round(K * a * a)
            x, _, y = curves[(round(lam), d2)]
            lab = rf"$\delta_2$ = {d2}"
            lab += " (normal base)" if d2 == 0 else ""
            ax.plot(x, y, lw=1.4, label=lab)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("u")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel(r"density $f_U(u)$")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"figure saved to {args.out}")


if __name__ == "__main__":
    main()
