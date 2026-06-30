"""Validation of a diagonal third-cumulant (skewness) correction explored as
an extension to the estimator in this repository. This correction is NOT
part of the live submission (estimator.py); see README.md, section "Further
investigation: a diagonal third-cumulant correction" for why.

This script runs two checks against brute-force Monte Carlo:

Stage A: given the TRUE (sampled) mean, variance, and third cumulant of a
pre-activation, does a first-order Gram-Charlier correction predict the
post-ReLU mean, E[h^2], and E[h^3] more accurately than the plain Gaussian
formulas? This isolates the moment-correction formulas from the separate
question of how well a cheap (diagonal/independence) approximation
propagates the third cumulant between layers.

Stage B: does the full estimator (exact K=2 covariance propagation, plus a
diagonal-only third-cumulant correction propagated under an
independence approximation) reduce end-to-end final-layer MSE against a
low-noise Monte Carlo reference, relative to the K=2-only estimator?

Run: `python validate_skew_correction.py`. Requires only NumPy.
"""

import numpy as np

gl_x, gl_w = np.polynomial.legendre.leggauss(24)


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def norm_cdf(x):
    _P = 0.2316419
    _A1, _A2, _A3 = 0.319381530, -0.356563782, 1.781477937
    _A4, _A5 = -1.821255978, 1.330274429
    t = 1.0 / (1.0 + _P * np.abs(x))
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    pdf = norm_pdf(x)
    cdf = 1.0 - pdf * poly
    return np.where(x >= 0, cdf, 1.0 - cdf)


def relu_cov_formula(mu_x, mu_y, sigma_x, sigma_y, rho):
    """Exact Cov(ReLU(X), ReLU(Y)); see validate_relu_covariance.py."""
    a = mu_x / sigma_x
    b = mu_y / sigma_y
    rho = np.clip(rho, -1 + 1e-10, 1 - 1e-10)
    Phi_a = norm_cdf(a)
    Phi_b = norm_cdf(b)
    rho_e, a_e, b_e = rho[..., None], a[..., None], b[..., None]
    s = (rho_e / 2.0) * (gl_x + 1.0)
    one_minus_s2 = np.maximum(1.0 - s * s, 1e-12)
    f = (1.0 / (2 * np.pi * np.sqrt(one_minus_s2))) * np.exp(
        -(a_e * a_e - 2 * s * a_e * b_e + b_e * b_e) / (2 * one_minus_s2)
    )
    integrand = (rho_e - s) * f
    I = (rho / 2.0) * np.sum(gl_w * integrand, axis=-1)
    return sigma_x * sigma_y * (rho * Phi_a * Phi_b + I)


def gaussian_relu_moments(mu, sigma):
    """A = E[h], B = E[h^2], C = E[h^3] for h = ReLU(Z), Z ~ N(mu, sigma^2)."""
    alpha = mu / sigma
    Phi = norm_cdf(alpha)
    phi = norm_pdf(alpha)
    A = mu * Phi + sigma * phi
    B = (mu * mu + sigma * sigma) * Phi + mu * sigma * phi
    C = (mu**3 + 3 * mu * sigma**2) * Phi + (mu**2 * sigma + 2 * sigma**3) * phi
    return A, B, C, alpha, Phi, phi


def skew_corrected_moments(mu, sigma, kappa3):
    """First-order Gram-Charlier (skewness) correction to A, B, C above, and
    the implied kappa3 of h = ReLU(Z) via the moment-to-cumulant relation."""
    A0, B0, C0, alpha, Phi, phi = gaussian_relu_moments(mu, sigma)
    C1 = -(kappa3 * alpha) / (6 * sigma**2) * phi
    C2 = -(kappa3) / (6 * sigma) * phi
    C3 = kappa3 * (Phi / 2 - alpha * phi)
    A, B, C = A0 + C1, B0 + C2, C0 + C3
    kappa3_h = C - 3 * A * B + 2 * A**3
    return A, B, C, kappa3_h


def central_moments_from_samples(x):
    mu = x.mean(axis=0)
    c2 = ((x - mu) ** 2).mean(axis=0)
    c3 = ((x - mu) ** 3).mean(axis=0)
    return mu, c2, c3


def k2_layer_step(mu, cov, w):
    mu_pre = w.T @ mu
    cov_pre = w.T @ cov @ w
    var_pre = np.maximum(np.diag(cov_pre), 1e-12)
    sigma_pre = np.sqrt(var_pre)
    alpha = mu_pre / sigma_pre
    Phi, phi = norm_cdf(alpha), norm_pdf(alpha)

    mu_post = mu_pre * Phi + sigma_pre * phi
    ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma_pre * phi
    var_post = np.maximum(ez2 - mu_post**2, 0.0)

    n = w.shape[1]
    sigma_outer = np.outer(sigma_pre, sigma_pre)
    rho = np.clip(cov_pre / sigma_outer, -1 + 1e-7, 1 - 1e-7)
    cov_post = relu_cov_formula(
        mu_pre[:, None] * np.ones((1, n)), mu_pre[None, :] * np.ones((n, 1)),
        sigma_pre[:, None] * np.ones((1, n)), sigma_pre[None, :] * np.ones((n, 1)),
        rho,
    )
    np.fill_diagonal(cov_post, var_post)
    return mu_post, cov_post, mu_pre, sigma_pre, alpha, Phi, phi, var_pre


def run_k2(weights, width):
    mu, cov = np.zeros(width), np.eye(width)
    rows = []
    for w in weights:
        mu, cov, *_ = k2_layer_step(mu, cov, w)
        rows.append(mu.copy())
    return np.stack(rows, axis=0)


def run_k2_plus_diagonal_k3(weights, width):
    mu, cov, kappa3 = np.zeros(width), np.eye(width), np.zeros(width)
    rows = []
    for w in weights:
        mu_post_k2, cov_post, mu_pre, sigma_pre, alpha, Phi, phi, var_pre = k2_layer_step(mu, cov, w)
        kappa3_pre = (w**3).T @ kappa3  # diagonal/independence approximation
        A, B, C, kappa3_post = skew_corrected_moments(mu_pre, sigma_pre, kappa3_pre)
        mu, cov, kappa3 = A, cov_post, kappa3_post
        rows.append(mu.copy())
    return np.stack(rows, axis=0)


def monte_carlo_mean(weights, width, n_samples, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, width)).astype(np.float32)
    rows = []
    for w in weights:
        x = np.maximum(x @ w, 0.0)
        rows.append(x.mean(axis=0, dtype=np.float64))
    return np.stack(rows, axis=0)


def stage_a(width=16, depth=6, n_samples=8_000_000, seed=0):
    print("=== Stage A: marginal moment formulas, given TRUE kappa3 ===\n")
    rng = np.random.default_rng(seed)
    scale = (2.0 / width) ** 0.5
    weights = [(rng.standard_normal((width, width)) * scale).astype(np.float32) for _ in range(depth)]

    x = rng.standard_normal((n_samples, width)).astype(np.float32)
    pre = x
    for layer, w in enumerate(weights):
        z = pre @ w
        mu_true, var_true, kappa3_true = central_moments_from_samples(z)
        sigma_true = np.sqrt(var_true)
        h = np.maximum(z, 0.0)
        Eh_true = h.mean(axis=0)

        A0, _, _, _, _, _ = gaussian_relu_moments(mu_true, sigma_true)
        A, _, _, _ = skew_corrected_moments(mu_true, sigma_true, kappa3_true)

        err_gauss = np.abs(A0 - Eh_true).mean()
        err_skew = np.abs(A - Eh_true).mean()
        print(f"layer {layer}: mean|err| gaussian-only={err_gauss:.3e}  skew-corrected={err_skew:.3e}  "
              f"(ratio={err_skew / err_gauss:.3f})")
        pre = h
    print()


def stage_b(width, depth, n_mc, n_seeds, label):
    print(f"=== Stage B ({label}): end-to-end K2 vs K2+diagonal-K3, {n_seeds} seed(s) ===\n")
    batch = 2_000_000
    fmse_k2_all, fmse_k2k3_all, amse_k2_all, amse_k2k3_all = [], [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        scale = (2.0 / width) ** 0.5
        weights = [(rng.standard_normal((width, width)) * scale).astype(np.float32) for _ in range(depth)]

        pred_k2 = run_k2(weights, width)
        pred_k2k3 = run_k2_plus_diagonal_k3(weights, width)

        acc = np.zeros((depth, width), dtype=np.float64)
        n_done = 0
        for _ in range(n_mc // batch):
            acc += monte_carlo_mean(weights, width, batch, seed=1000 + seed) * batch
            n_done += batch
        ref = acc / n_done

        fmse_k2 = np.mean((pred_k2[-1] - ref[-1]) ** 2)
        fmse_k2k3 = np.mean((pred_k2k3[-1] - ref[-1]) ** 2)
        amse_k2 = np.mean((pred_k2 - ref) ** 2)
        amse_k2k3 = np.mean((pred_k2k3 - ref) ** 2)
        fmse_k2_all.append(fmse_k2)
        fmse_k2k3_all.append(fmse_k2k3)
        amse_k2_all.append(amse_k2)
        amse_k2k3_all.append(amse_k2k3)
        print(f"seed={seed}  final_mse: K2={fmse_k2:.4e} K2+K3={fmse_k2k3:.4e} (ratio={fmse_k2k3 / fmse_k2:.3f})  "
              f"all_mse: K2={amse_k2:.4e} K2+K3={amse_k2k3:.4e} (ratio={amse_k2k3 / amse_k2:.3f})")
    # Ratio of means across seeds (not mean of per-seed ratios): this avoids
    # over-weighting a seed whose K2 baseline MSE happens to be unusually
    # small, and is the statistic quoted in README.md.
    ratio_final = np.mean(fmse_k2k3_all) / np.mean(fmse_k2_all)
    ratio_all = np.mean(amse_k2k3_all) / np.mean(amse_k2_all)
    print(f"\nRatio of mean MSE across seeds (K2+K3 / K2), lower than 1.0 means K2+K3 has "
          f"lower MSE: final_layer={ratio_final:.3f}  all_layers={ratio_all:.3f}\n")


if __name__ == "__main__":
    stage_a(width=16, depth=6, n_samples=8_000_000)
    stage_b(width=16, depth=6, n_mc=30_000_000, n_seeds=5, label="small scale")
    # 3 seeds at the competition shape takes several minutes (each seed
    # forward-propagates 4M Monte Carlo samples through a 256x32 network in
    # plain NumPy); this is the same run that produced the README's
    # competition-scale numbers (ratio 0.824 for final-layer, 0.810 for
    # all-layers).
    stage_b(width=256, depth=32, n_mc=4_000_000, n_seeds=3, label="competition scale, width=256 depth=32")
