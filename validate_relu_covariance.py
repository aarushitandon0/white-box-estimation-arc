"""Standalone validation of the exact Cov(ReLU(X), ReLU(Y)) formula used in
`estimator.py`, checked against brute-force Monte Carlo.

For jointly Gaussian (X, Y) with means mu_x, mu_y, std devs sigma_x, sigma_y,
and correlation rho, the formula computes

    Cov(ReLU(X), ReLU(Y)) = sigma_x * sigma_y * [rho * Phi(a) * Phi(b) + I(rho, a, b)]

where a = mu_x/sigma_x, b = mu_y/sigma_y, and I is a 1-D Gauss-Legendre
quadrature (see README.md for the full derivation via Price's theorem).

Run: `python validate_relu_covariance.py`. Requires only numpy.
"""

import numpy as np

gl_x, gl_w = np.polynomial.legendre.leggauss(24)


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def norm_cdf(x):
    # Abramowitz & Stegun 26.2.17 rational approximation (accurate to <7.5e-8),
    # matching what's allowed inside the flopscope sandbox (no scipy there).
    _P = 0.2316419
    _A1, _A2, _A3 = 0.319381530, -0.356563782, 1.781477937
    _A4, _A5 = -1.821255978, 1.330274429
    t = 1.0 / (1.0 + _P * np.abs(x))
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    pdf = norm_pdf(x)
    cdf = 1.0 - pdf * poly
    return np.where(x >= 0, cdf, 1.0 - cdf)


def relu_cov_formula(mu_x, mu_y, sigma_x, sigma_y, rho):
    """Cov(ReLU(X), ReLU(Y)) via the closed-form-via-quadrature derivation."""
    a = mu_x / sigma_x
    b = mu_y / sigma_y
    rho = np.clip(rho, -1 + 1e-10, 1 - 1e-10)

    Phi_a = norm_cdf(a)
    Phi_b = norm_cdf(b)

    s = (rho / 2.0) * (gl_x + 1.0)
    one_minus_s2 = np.maximum(1.0 - s * s, 1e-12)
    f = (1.0 / (2 * np.pi * np.sqrt(one_minus_s2))) * np.exp(
        -(a * a - 2 * s * a * b + b * b) / (2 * one_minus_s2)
    )
    integrand = (rho - s) * f
    I = (rho / 2.0) * np.sum(gl_w * integrand, axis=-1)

    return sigma_x * sigma_y * (rho * Phi_a * Phi_b + I)


def relu_cov_montecarlo(mu_x, mu_y, sigma_x, sigma_y, rho, n=20_000_000, seed=1):
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x = mu_x + sigma_x * z1
    y = mu_y + sigma_y * (rho * z1 + np.sqrt(1 - rho * rho) * z2)
    rx = np.maximum(x, 0.0)
    ry = np.maximum(y, 0.0)
    return np.cov(rx, ry)[0, 1]


CASES = [
    (0.0, 0.0, 1.0, 1.0, 0.5),
    (0.0, 0.0, 1.0, 1.0, -0.5),
    (1.0, -1.0, 1.0, 1.0, 0.3),
    (2.0, 0.5, 1.5, 0.8, 0.7),
    (-1.0, -2.0, 1.0, 1.0, -0.6),
    (0.3, 0.3, 2.0, 2.0, 0.9),
    (0.0, 0.0, 1.0, 1.0, 0.0),
    (5.0, -5.0, 1.0, 1.0, 0.99),
    (0.1, 0.1, 0.5, 0.5, -0.95),
]

if __name__ == "__main__":
    print(f"{'mu_x':>6} {'mu_y':>6} {'sx':>5} {'sy':>5} {'rho':>6} | {'formula':>12} {'MC (20M)':>12} {'abs_diff':>10}")
    for mu_x, mu_y, sx, sy, rho in CASES:
        formula = relu_cov_formula(mu_x, mu_y, sx, sy, rho)
        mc = relu_cov_montecarlo(mu_x, mu_y, sx, sy, rho)
        print(f"{mu_x:6.2f} {mu_y:6.2f} {sx:5.2f} {sy:5.2f} {rho:6.2f} | {formula:12.6f} {mc:12.6f} {abs(formula - mc):10.2e}")
