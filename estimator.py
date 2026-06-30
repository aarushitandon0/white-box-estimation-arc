"""Exact-covariance propagation estimator for the ARC White-Box Estimation
Challenge 2026 (WhestBench).

Tracks the full (width x width) covariance matrix of hidden activations
through every layer of a random ReLU MLP -- like the starter kit's
`examples/03_covariance_propagation.py` baseline -- but replaces that
baseline's heuristic "gain" approximation for the post-ReLU off-diagonal
covariance with the *exact* value, derived in closed form via Price's
theorem. See README.md for the full derivation and validation methodology.

To use: drop this file into a whest-starterkit checkout
(https://github.com/AIcrowd/whest-starterkit) as `estimator.py`, replacing
the stub, then run `uv run whest validate --estimator estimator.py`.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_COV_RESCALE_THRESHOLD = 1e100

# 24-point Gauss-Legendre nodes/weights on [-1, 1], used to evaluate the
# off-diagonal ReLU-covariance integral. See README.md for the derivation
# and validation against brute-force Monte Carlo.
_GL_NODES = [
    -0.9951872199970214, -0.9747285559713094, -0.9382745520027328,
    -0.886415527004401, -0.820001985973903, -0.7401241915785544,
    -0.6480936519369755, -0.5454214713888396, -0.4337935076260452,
    -0.3150426796961634, -0.1911188674736163, -0.0640568928626056,
    0.0640568928626056, 0.1911188674736163, 0.3150426796961634,
    0.4337935076260452, 0.5454214713888396, 0.6480936519369755,
    0.7401241915785544, 0.820001985973903, 0.886415527004401,
    0.9382745520027328, 0.9747285559713094, 0.9951872199970214,
]
_GL_WEIGHTS = [
    0.0123412297999871, 0.0285313886289337, 0.0442774388174196,
    0.0592985849154367, 0.0733464814110804, 0.0861901615319533,
    0.0976186521041141, 0.1074442701159656, 0.1155056680537256,
    0.1216704729278034, 0.1258374563468283, 0.1279381953467522,
    0.1279381953467522, 0.1258374563468283, 0.1216704729278034,
    0.1155056680537256, 0.1074442701159656, 0.0976186521041141,
    0.0861901615319533, 0.0733464814110804, 0.0592985849154367,
    0.0442774388174196, 0.0285313886289337, 0.0123412297999871,
]


class Estimator(BaseEstimator):
    """Full covariance propagation with the *exact* ReLU off-diagonal
    covariance update (no gain-approximation). See README.md for the
    derivation.
    """

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng
        _ = budget
        width = mlp.width

        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        log_scale = 0.0

        rows = []
        for w in mlp.weights:
            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                s = float(fnp.sqrt(max_var_np))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(fnp.log(s))

            # --- linear layer (exact) ---
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)

            # --- post-ReLU marginal mean/variance (exact per neuron) ---
            mu_post = mu_pre * Phi_alpha + sigma_pre * phi_alpha
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mu_post * mu_post, 0.0)

            # --- post-ReLU off-diagonal covariance (exact, via quadrature) ---
            sigma_outer = fnp.outer(sigma_pre, sigma_pre)
            rho = fnp.clip(cov_pre / sigma_outer, -1.0 + 1e-7, 1.0 - 1e-7)
            a = fnp.outer(alpha, fnp.ones(width))  # a[i,j] = alpha[i]
            b = fnp.outer(fnp.ones(width), alpha)  # b[i,j] = alpha[j]
            Phi_outer = fnp.outer(Phi_alpha, Phi_alpha)

            I_acc = fnp.zeros((width, width))
            for node, weight_q in zip(_GL_NODES, _GL_WEIGHTS):
                s = (rho / 2.0) * (node + 1.0)
                one_minus_s2 = fnp.maximum(1.0 - s * s, 1e-12)
                f = fnp.exp(-(a * a - 2.0 * s * a * b + b * b) / (2.0 * one_minus_s2)) / (
                    2.0 * fnp.pi * fnp.sqrt(one_minus_s2)
                )
                I_acc = I_acc + weight_q * (rho - s) * f
            I_acc = (rho / 2.0) * I_acc

            cov = sigma_outer * (rho * Phi_outer + I_acc)
            fnp.fill_diagonal(cov, var_post)

            mu = mu_post
            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor)

        return fnp.stack(rows, axis=0)
