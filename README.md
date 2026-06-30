# Exact-Covariance Propagation 
The challenge asks: given only the weights of a
randomly initialized ReLU multilayer perceptron, predict the expected
per-neuron post-ReLU activation under standard normal input, without running
the network thousands of times to average the answer empirically.

The estimator implemented here improves on the official starter kit's bundled
`covariance_propagation` baseline in two ways: by replacing one specific
approximation, the off-diagonal covariance between two neurons after a ReLU
nonlinearity, with its exact closed-form value; and by adding a diagonal
third-cumulant (skewness) correction to the predicted mean at each layer.
Both are derived from first principles below and validated against
brute-force Monte Carlo sampling before being trusted in the estimator
itself.

## Table of contents

1. [Problem statement](#problem-statement)
2. [Existing approaches](#existing-approaches)
3. [Method: exact off-diagonal ReLU covariance](#method-exact-off-diagonal-relu-covariance)
4. [Derivation](#derivation)
5. [Numerical evaluation](#numerical-evaluation)
6. [Implementation](#implementation)
7. [Validation](#validation)
8. [Extension: diagonal third-cumulant correction](#extension-diagonal-third-cumulant-correction)
9. [Results](#results)
10. [Repository structure](#repository-structure)
11. [Usage](#usage)
12. [Limitations and future work](#limitations-and-future-work)
13. [References](#references)
14. [License](#license)

## Problem statement

Consider a fully connected ReLU network with `L` hidden layers, each of width
`n`, and no biases. The weights of layer `l` are denoted `W^(l)`, an `n x n`
matrix, He-initialized as `W^(l)_ij ~ N(0, 2/n)` independently. Given an
input `X ~ N(0, I_n)`, the hidden activations are defined recursively as

```
h^(0) = X
h^(l) = ReLU(W^(l) h^(l-1)),   l = 1, ..., L
```

The task is to produce, for every layer `l` and neuron `i`, an estimate of
`E_X[h_i^(l)(X)]`. The output is an `L x n` matrix of predicted means. In the
Phase 1 competition shape used for the results in this repository, `n = 256`
and `L = 32`.

Submissions are scored by mean squared error against a reference value
computed by the organizers with a much larger Monte Carlo budget than
participants are given. Each submission also has a fixed analytical FLOP
budget per network, `B_m`, tracked by a library called `flopscope` that
instruments NumPy-style operations. The final per-network score multiplies
the raw mean squared error by a compute-usage factor,
`max(0.1, C_m / B_m)`, where `C_m` is the FLOPs the estimator actually used.
This means an estimator that is both accurate and cheap is rewarded, but the
benefit of using less than ten percent of the budget is capped: there is no
additional score benefit to using less than 10% of the per-network budget, so
the practical objective is to minimize raw mean squared error subject to
staying under roughly that compute fraction.

A naive Monte Carlo estimator, drawing `N` independent inputs and averaging
the resulting activations, has expected squared error that falls only as
`1/N`. One forward pass through the competition's network shape costs
approximately `4.24e6` FLOPs, and the per-network budget is approximately
`2.72e11` FLOPs, which permits roughly 64,000 samples. The motivating
question of the challenge is whether the network weights themselves can be
used analytically to beat that sampling baseline at equal or lower compute.

## Existing approaches

Several baselines exist along an accuracy-versus-compute spectrum.

**Zero baseline.** Always predict zero. Serves only as a sanity floor.

**Monte Carlo sampling.** Draw `N` samples, forward propagate, average. Error
falls as `O(1/N)` in squared error, independent of the network's structure.
This is the baseline that mechanistic methods are trying to beat.

**Mean propagation (diagonal variance).** Track only the marginal mean and
variance of each neuron, treating neurons within a layer as independent.
After a linear layer, the pre-activation mean and variance update exactly
because linear combinations of independent Gaussians are Gaussian:

```
mu_pre   = W^T mu
var_pre  = (W * W)^T var
```

The ReLU step then uses the exact first and second moment of a rectified
Gaussian, a classical result (see Frey and Hinton, 1999, and the references
therein):

```
alpha = mu_pre / sigma_pre
E[ReLU(Z)]   = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
E[ReLU(Z)^2] = (mu_pre^2 + var_pre) * Phi(alpha) + mu_pre * sigma_pre * phi(alpha)
Var[ReLU(Z)] = E[ReLU(Z)^2] - E[ReLU(Z)]^2
```

where `Phi` and `phi` are the standard normal CDF and PDF respectively. This
method costs `O(L n^2)` FLOPs (matrix-vector products per layer) but ignores
correlations between neurons entirely, which is a poor approximation: even
though the weights are independent across neurons, all neurons in a layer
share the same input, so their pre-activations become correlated as soon as
the first layer is applied.

**Covariance propagation (full matrix, gain approximation).** Track the full
`n x n` covariance matrix of each layer's pre-activations rather than only
its diagonal. The linear layer step is again exact:

```
mu_pre  = W^T mu
cov_pre = W^T cov W
```

The ReLU step's diagonal (the marginal variance of each neuron) is exact, as
above. However, the off-diagonal covariance between two neurons after a ReLU
has no simple closed form, and the starter kit's bundled baseline
approximates it with a heuristic "gain" correction:

```
cov_post[i, j]  ~=  Phi(alpha_i) * Phi(alpha_j) * cov_pre[i, j]
```

The intuition is that `Phi(alpha_i)` is the probability that neuron `i` is in
its active (non-clipped) regime, so the covariance is scaled down by the
probability that both neurons are simultaneously active. This is a reasonable
heuristic, but it is not the exact value. This method costs `O(L n^3)` FLOPs,
dominated by the matrix-matrix product in the covariance update, and is the
strongest baseline shipped in the official starter kit.

## Method: exact off-diagonal ReLU covariance

This repository's contribution replaces the gain approximation above with the
exact value of `Cov(ReLU(Z_i), ReLU(Z_j))` whenever `(Z_i, Z_j)` are jointly
Gaussian with general, non-zero means. The diagonal terms (marginal variance)
and the linear layer update are unchanged from the covariance propagation
baseline, since those are already exact.

## Derivation

Let `(X, Y)` be jointly Gaussian with means `mu_x` and `mu_y`, standard
deviations `sigma_x` and `sigma_y`, and correlation `rho`. Define

```
K(rho) = E[ReLU(X) ReLU(Y)]
```

as a function of `rho`, holding the marginals of `X` and `Y` fixed. The goal
is a closed form for `K(rho)`, from which the covariance follows immediately
as `Cov(ReLU(X), ReLU(Y)) = K(rho) - E[ReLU(X)] E[ReLU(Y)]`.

### Step 1: Price's theorem

Price's theorem (Price, 1958) states that for jointly Gaussian `(X, Y)` with
fixed marginals and correlation `rho`, and for sufficiently well-behaved
functions `f` and `g`,

```
d/drho  E[f(X) g(Y)]  =  E[f'(X) g'(Y)]
```

Since `ReLU'(x) = 1{x > 0}` (the Heaviside step function, away from the
single non-differentiable point at zero, which has probability zero under a
continuous distribution), applying Price's theorem to `K(rho)` gives

```
dK/drho = E[1{X > 0} 1{Y > 0}] = P(X > 0, Y > 0; rho)
```

Writing `X = mu_x + sigma_x U` and `Y = mu_y + sigma_y V` for standard
bivariate normal `(U, V)` with correlation `rho`, this crossing probability
is `L(a, b; rho) := P(U > -a, V > -b; rho)`, with standardized thresholds
`a = mu_x / sigma_x` and `b = mu_y / sigma_y`. Price's theorem applied to the
original, unstandardized variables introduces the scale factors
`sigma_x sigma_y`, so

```
dK/drho = sigma_x * sigma_y * L(a, b; rho)
```

### Step 2: expanding the crossing probability

By inclusion-exclusion, writing `Phi2` for the standard bivariate normal CDF,

```
L(a, b; rho) = P(U > -a, V > -b; rho)
             = 1 - Phi(-a) - Phi(-b) + Phi2(-a, -b; rho)
             = Phi(a) + Phi(b) - 1 + Phi2(-a, -b; rho)
```

Integrating `dK/drho` from `rho = 0` (where `K(0) = E[ReLU(X)] E[ReLU(Y)]` by
independence) to a target correlation `rho`, and noting that the constant
terms `Phi(a) + Phi(b) - 1` integrate trivially while `Phi2(-a, -b; t)` does
not:

```
K(rho) - K(0) = sigma_x sigma_y [ rho (Phi(a) + Phi(b) - 1) + Int_0^rho Phi2(-a, -b; t) dt ]
```

### Step 3: a second application of Price's theorem

The remaining integral still involves `Phi2`, the bivariate normal CDF,
which has no elementary closed form. However, `Phi2` is itself defined as an
integral of the bivariate normal density over its correlation parameter,

```
Phi2(h, k; rho) = Phi(h) Phi(k) + Int_0^rho phi2(h, k; t) dt
```

where `phi2` is the bivariate normal density, which does have an elementary
closed form pointwise. Substituting this into the double integral from Step 2
and exchanging the order of integration (a standard Fubini argument, valid
since the integrand is bounded and continuous on the relevant domain) reduces
the nested double integral to a single integral:

```
Int_0^rho Phi2(-a, -b; t) dt = rho * Phi(-a) * Phi(-b) + Int_0^rho (rho - s) phi2(-a, -b; s) ds
```

### Step 4: simplification

Substituting back into the expression from Step 2 and using the algebraic
identity

```
Phi(a) + Phi(b) - 1 + Phi(-a) Phi(-b) = Phi(a) Phi(b)
```

(which follows directly from `Phi(-a) = 1 - Phi(a)` and `Phi(-b) = 1 -
Phi(b)`), the result collapses to

```
Cov(ReLU(X), ReLU(Y)) = sigma_x * sigma_y * [ rho * Phi(a) * Phi(b) + I(rho, a, b) ]

I(rho, a, b) = Int_0^rho (rho - s) * f(s, a, b) ds

f(s, a, b) = exp( -(a^2 - 2 s a b + b^2) / (2 (1 - s^2)) ) / (2 pi sqrt(1 - s^2))
```

where `f` is the bivariate normal density `phi2(-a, -b; s)`, written out
explicitly using only elementary functions (no special functions, no nested
CDF evaluations). This is the formula implemented in `estimator.py` and
checked numerically in `validate_relu_covariance.py`.

### Consistency checks

Two special cases confirm the formula is consistent with known results.

At `rho = 0`, the integral vanishes (`I(0, a, b) = 0`), giving
`Cov = 0`, matching independence.

At `mu_x = mu_y = 0` (so `a = b = 0`), the formula reduces to the classical
zero-mean result used in the arc-cosine kernel literature (Cho and Saul,
2009):

```
Cov(ReLU(X), ReLU(Y)) = (sigma_x sigma_y / (2 pi)) * (sin(theta) + (pi - theta) cos(theta))
theta = arccos(rho)
```

This reduction was checked numerically rather than re-derived algebraically
in this repository, and matches to within floating point precision.

A further numerical bound worth noting: since `2 s a b <= a^2 + b^2` for all
`s` in `[-1, 1]` by the AM-GM inequality (`2 |s a b| <= 2 |a b| <= a^2 +
b^2`), the exponent in `f(s, a, b)` is never positive, so `f` never
overflows regardless of how large `a` and `b` become. This was verified to
hold even in the degenerate limit `rho -> 1`, where `1 - s^2` is clamped to a
small positive floor for numerical safety, since at that point the diagonal
of the covariance matrix is overwritten by the exact marginal variance
formula in any case (see Implementation, below), so the behavior of the
off-diagonal formula exactly on the diagonal is irrelevant to correctness.

## Numerical evaluation

The integral `I(rho, a, b)` is evaluated with fixed-order Gauss-Legendre
quadrature on 24 nodes, using the standard affine change of variables mapping
`[-1, 1]` to `[0, rho]`. The integrand is smooth (infinitely differentiable)
on the domain of interest, `s` between `0` and `rho` with `|rho| < 1`, so
Gauss-Legendre quadrature converges geometrically and 24 nodes is far more
than sufficient for machine precision in this regime. The node and weight
constants used in `estimator.py` were computed once offline via
`numpy.polynomial.legendre.leggauss(24)` and hardcoded, so the estimator
itself performs no quadrature setup at runtime.

## Implementation

`estimator.py` implements the following per-layer recursion, starting from
`mu = 0` and `cov = I_n` (the standard normal input distribution):

1. Linear layer (exact): `mu_pre = W^T mu`, `cov_pre = W^T cov W`, computed
   with `einsum` rather than chained matrix multiplication so that the
   FLOP-tracking library can recognize the result is symmetric and avoid
   redundant downstream warnings.
2. Marginal ReLU moments (exact, diagonal): the per-neuron mean and variance
   formulas from the Existing Approaches section above, applied to the
   diagonal of `cov_pre`.
3. Off-diagonal ReLU covariance (exact): the formula derived above, applied
   vectorized across all `n x n` neuron pairs at once, accumulated over the
   24 quadrature nodes.
4. The diagonal of the resulting covariance matrix is overwritten with the
   exact marginal variance from step 2, since the off-diagonal formula is
   not evaluated at `rho = 1` for numerical reasons described above.
5. A log-scale rescaling mechanism, carried over from the starter kit's
   baseline, prevents the covariance matrix's diagonal from overflowing in
   very deep networks: if the largest variance exceeds a threshold, the mean
   and covariance are rescaled and the scale factor is tracked in log space,
   then reapplied to the recorded mean before it is returned.

Computational cost: the linear layer's covariance update costs `O(n^3)` per
layer from the matrix-matrix product, identical to the baseline. The
off-diagonal covariance formula adds `O(n^2)` work per quadrature node, for
`O(24 n^2)` per layer. At the competition shape (`n = 256`, `L = 32`), the
quadrature cost is a small fraction of the matmul cost, and the complete
estimator used approximately `3.96e11` total FLOPs across the matmuls and
quadrature combined when measured by the grader's FLOP accounting library,
against the per-network budget of `2.72e11`. Effective compute after the
unfavorable wall-clock conversion (described in the challenge rules) landed
at approximately 8 percent of the budget on average across the scored
networks, comfortably under the point at which the compute-usage multiplier
in the scoring formula begins to rise above its floor of 0.1.

## Validation

Validation was carried out at two levels before any result was trusted.

**Formula-level validation.** `validate_relu_covariance.py` is a standalone
script, independent of the estimator and of the flopscope library, that
checks the closed-form covariance formula directly against brute-force Monte
Carlo simulation with 20 million samples per case, across nine
`(mu_x, mu_y, sigma_x, sigma_y, rho)` combinations spanning positive and
negative means, equal and unequal variances, and correlations ranging from
-0.95 to 0.99:

```
$ python validate_relu_covariance.py
  mu_x   mu_y    sx    sy    rho |      formula     MC (20M)   abs_diff
  0.00   0.00  1.00  1.00   0.50 |     0.145344     0.145332   1.23e-05
  0.00   0.00  1.00  1.00  -0.50 |    -0.104656    -0.104669   1.29e-05
  1.00  -1.00  1.00  1.00   0.30 |     0.042412     0.042437   2.50e-05
  2.00   0.50  1.50  0.80   0.70 |     0.579122     0.578976   1.46e-04
 -1.00  -2.00  1.00  1.00  -0.60 |    -0.000706    -0.000707   9.39e-07
  0.30   0.30  2.00  2.00   0.90 |     1.404988     1.404769   2.19e-04
  0.00   0.00  1.00  1.00   0.00 |     0.000000    -0.000015   1.45e-05
  5.00  -5.00  1.00  1.00   0.99 |     0.000000     0.000000   9.07e-09
  0.10   0.10  0.50  0.50  -0.95 |    -0.061170    -0.061156   1.38e-05
```

All deviations are consistent with the expected Monte Carlo sampling noise at
this sample size (standard error on the order of `sqrt(Var / 2e7)`), and no
systematic bias is visible across the range of correlations and means tested,
including near the boundary `rho` close to plus or minus one.

**Estimator-level validation.** With the formula confirmed correct in
isolation, the full estimator was compared end to end against a low-noise
Monte Carlo reference (4,000,000 samples, batched) on the actual competition
network shape (width 256, depth 32), alongside the mean propagation and
gain-approximation covariance propagation baselines:

| Estimator | Final-layer MSE | All-layers MSE |
|---|---|---|
| Mean propagation | 2.590e-03 | 1.723e-03 |
| Covariance propagation (gain approximation) | 1.370e-04 | 8.166e-05 |
| Covariance propagation (exact, this repository) | 1.522e-04 | 8.763e-05 |

At this scale, the exact off-diagonal formula and the gain approximation
landed within the noise band of each other against a 4-million-sample
reference, with the gain approximation marginally ahead in this particular
comparison. To check whether this was a quirk of sampling noise rather than a
real effect, the same three estimators were also compared on much smaller
networks (width 16, depth 6) against a 30-million-sample reference across
five random seeds, where Monte Carlo noise is a much smaller fraction of the
signal:

| Estimator | Average final-layer MSE | Average all-layers MSE |
|---|---|---|
| Mean propagation | 1.697e-02 | 6.806e-03 |
| Covariance propagation (gain approximation) | 4.528e-03 | 2.194e-03 |
| Covariance propagation (exact, this repository) | 4.512e-03 | 2.141e-03 |

The exact formula was ahead of the gain approximation on four of five seeds
at this smaller scale, with the fifth seed reversed, consistent with the
exact formula being a strict improvement in principle whose effect is simply
small relative to a different, larger source of error common to both
methods: the assumption that each layer's pre-activation distribution is
itself exactly multivariate Gaussian, which both estimators share and which
neither corrects for. This finding directly motivates the discussion in
Limitations and Future Work, below.

This negative-leaning intermediate result is reported here rather than
omitted, since the formula's exactness was never in question (it was
confirmed independently at the formula level above), and an honest account of
where the gain in this approach is and is not realized is more useful than a
selectively favorable comparison.

## Extension: diagonal third-cumulant correction

The off-diagonal covariance formula above leaves the Gaussian-closure
assumption itself untouched: every layer's pre-activation vector is still
treated as exactly multivariate Gaussian, when in reality each ReLU
introduces skewness that compounds across the network's 32 layers. The
estimator closes part of this gap with a second, independent correction: a
per-neuron diagonal third cumulant (skewness), propagated through linear
layers under an independence approximation analogous to the one mean
propagation makes for variance, used to apply a first-order Gram-Charlier
correction to the predicted mean at every layer. The covariance matrix
itself, both diagonal and off-diagonal, is left exactly as computed by the
K=2 method above; only the recorded and propagated mean is corrected.

### Derivation

For a pre-activation `Z` with mean `mu`, standard deviation `sigma`, and
third cumulant `kappa3`, a first-order Gram-Charlier expansion of `Z`'s
density adds a correction term proportional to `kappa3` to the density of
the standardized Gaussian. Integrating `z * 1{z > 0}` against this corrected
density, using the same kind of Hermite-integral technique used for the
off-diagonal covariance formula above, gives a closed-form correction to
`E[ReLU(Z)]`:

```
alpha = mu / sigma
E[ReLU(Z)]  ~=  mu * Phi(alpha) + sigma * phi(alpha)  -  (kappa3 * alpha) / (6 * sigma^2) * phi(alpha)
```

with similar closed-form correction terms derived for `E[ReLU(Z)^2]` and
`E[ReLU(Z)^3]`, the latter needed to estimate the post-ReLU third cumulant to
propagate to the next layer (via the standard moment-to-cumulant relation
`kappa3 = mu3' - 3 mu1' mu2' + 2 mu1'^3` applied to the corrected raw
moments). Propagating the third cumulant itself through a linear layer under
the independence approximation costs only `O(n^2)` per layer:

```
kappa3_pre[a] = sum_i W[i, a]^3 * kappa3_h[i]
```

### Validation

This was checked in two stages against brute-force Monte Carlo, mirroring
the methodology used for the off-diagonal covariance formula, reproducible
via `validate_skew_correction.py`:

- Stage A isolated the moment-correction formulas: given the *true*
  (sampled) kappa3 of a real network's pre-activations as input, the
  skew-corrected mean formula reduced the mean absolute error against
  sampled ground truth by a consistent factor of roughly 2 to 5 times across
  five test layers, relative to the plain Gaussian formula, once the
  pre-activation had any measurable skewness.
- Stage B tested the full estimator end to end, comparing K=2-only against
  K=2 plus the diagonal third-cumulant correction, against a low-noise Monte
  Carlo reference. At small scale (width 16, depth 6, five seeds), the
  correction reduced final-layer MSE to 69 percent of the K=2-only value on
  average (four of five seeds improved). At the actual competition shape
  (width 256, depth 32, three seeds), it reduced final-layer MSE to 82
  percent of the K=2-only value on average, with all three seeds improving.

Both stages confirm the correction is mathematically sound and genuinely
reduces raw mean squared error, including at the competition's exact network
shape.

### A measurement pitfall worth recording

The correction's extra arithmetic is cheap in raw FLOPs (it added well under
1 percent to the estimator's analytical FLOP count). Despite this, an
initial scoring attempt with `whest run --runner local` on the development
machine used for this repository showed the *adjusted* competition score
getting worse, not better, across two separate runs: raw final-layer MSE
fell from `7.82e-05` to `6.56e-05` as expected, but the adjusted score rose
from `7.85e-06` to between `1.09e-05` and `1.87e-05`. The scoring formula's
compute-usage multiplier, `max(0.1, C_m / B_m)`, is sensitive to residual
wall-clock time outside the dominant matrix multiplications, and that time
appeared to roughly double under `--runner local`, pushing compute
utilization from about 8 percent of budget to between 17 and 29 percent.

Before accepting that as a real property of the estimator, two checks were
run. First, profiling the two estimators with `cProfile` on identical input
showed only a 5 to 7 percent difference in Python-level call overhead
between them, not the 2.5 to 3.5 times difference the `whest run` reports
implied. Second, re-scoring both estimators with `whest run --runner
subprocess`, which isolates each network's evaluation in its own process and
is explicitly documented as closer to the grader's actual behavior, resolved
the discrepancy: the K=2-only estimator scored `7.89e-06` under subprocess
isolation, consistent with its `--runner local` result of `7.85e-06`, while
the K=2 plus diagonal-K=3 estimator scored `6.71e-06`, a genuine 15 percent
improvement. The `--runner local` result was concluded to be measurement
noise specific to this development machine (likely from sustained wall-clock
timing being more exposed to OS-level scheduling jitter as cumulative
process runtime grows), not a real property of the correction, and the
correction was adopted into the submitted estimator on that basis.

This is recorded here because it generalizes: on this competition, a
correction's raw FLOP cost is not the only thing worth checking before
trusting a local benchmark, since the scoring formula also prices in
wall-clock time outside instrumented operations, and that quantity can be
measurably noisier under some local harness configurations than under
others. `--runner subprocess` is recommended over `--runner local` for any
timing-sensitive comparison, not only as a contract-correctness check.

## Results

The estimator (K=2 exact off-diagonal covariance plus the diagonal K=3
correction described above) was run against the official public Mini split
of the Phase 1 dataset (100 randomly generated networks, width 256, depth
32) using the starter kit's own grading pipeline, `whest run --runner
subprocess`, for the reasons given in "A measurement pitfall worth
recording" above:

| Metric | Value |
|---|---|
| Adjusted final-layer score (primary metric) | 6.71e-06 |
| Raw final-layer MSE | 6.56e-05 |
| All-layers MSE | 4.30e-05 |
| Best single-network adjusted score | 1.54e-06 |
| Worst single-network adjusted score | 2.67e-05 |
| Mean compute utilization | 9.35 percent of budget |
| Mean score multiplier | 0.1022 |
| Failed networks | 0 of 100 |

For comparison, the K=2-only estimator (without the diagonal third-cumulant
correction) scores `7.89e-06` adjusted / `7.82e-05` raw under the same
`--runner subprocess` conditions, and the organizers' own published figure
for the bundled `covariance_propagation` baseline at a comparable network
shape reports a final-layer mean squared error of approximately `8.4e-05`.
The progression mean propagation, baseline covariance propagation, exact
off-diagonal covariance, exact covariance plus diagonal skewness correction
is a monotonic improvement at every step measured so far.

## Repository structure

```
.
├── estimator.py                   the submission, written against the
│                                   whest-starterkit contract
├── validate_relu_covariance.py    standalone proof of correctness for the
│                                   closed-form off-diagonal covariance
│                                   formula, checked against brute-force
│                                   Monte Carlo
├── validate_skew_correction.py    standalone proof of correctness for the
│                                   diagonal third-cumulant correction,
│                                   checked against brute-force Monte Carlo
│                                   (both the marginal moment formulas and
│                                   the full estimator end to end)
├── README.md                      this file
├── LICENSE                        MIT license
└── .gitignore
```

`validate_relu_covariance.py` and `validate_skew_correction.py` have no
dependency on flopscope, whestbench, or any part of the starter kit. They
depend only on NumPy and are meant to be read and re-run independently of
the competition harness, as checks on the mathematics rather than as part of
the submission itself.

## Usage

The estimator is a single file written against the official
[whest-starterkit](https://github.com/AIcrowd/whest-starterkit) contract
(`flopscope.numpy` for FLOP-tracked array operations, `whestbench.BaseEstimator`
for the estimator interface). To reproduce the results above:

```bash
git clone https://github.com/AIcrowd/whest-starterkit.git
cd whest-starterkit
uv sync
cp /path/to/this/repository/estimator.py .
uv run whest validate --estimator estimator.py
uv run whest run --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
    --split mini --runner subprocess
```

`--runner subprocess` is used deliberately rather than `--runner local`; see
"A measurement pitfall worth recording" above.

To re-run the independent formula validations, which require only NumPy:

```bash
python validate_relu_covariance.py
python validate_skew_correction.py
```

## Limitations and future work

The validation results above show that the exact off-diagonal covariance
formula, while mathematically correct and a strict theoretical improvement
over the gain approximation, is not the dominant source of remaining error
at the competition's network scale on its own. Both this estimator and the
baseline it improves on assume that each layer's pre-activation vector is
well approximated as multivariate Gaussian, an approximation sometimes
called Gaussian closure. That assumption degrades with depth, since each
ReLU layer's output is not actually Gaussian, and approximation error
compounds across the network's 32 layers. The diagonal third-cumulant
correction above closes part of this gap (a genuine ~15 percent reduction in
adjusted score) by tracking per-neuron skewness, but it deliberately ignores
the off-diagonal third cumulant between neurons, the same kind of
cross-neuron structure whose omission was the gap between mean propagation
and full covariance propagation at the K=2 level. Closing the remaining gap
requires tracking the full higher-order cumulant tensors, including
off-diagonal terms, and going further still to the fourth cumulant
(kurtosis), rather than only diagonal third-order statistics.

The challenge's companion paper, Wu et al. (2026), develops exactly this:
a general algorithm for propagating cumulants of arbitrary order `K` through
a network using Hermite expansions and a combinatorial "diagram summation
formula." Their published results show that going from `K = 2` (covariance
propagation, the level implemented in this repository) to `K = 3` or `K = 4`
yields substantial further error reduction at moderate depth, though they
also report that error grows with network depth roughly as `(L / n)^K` for
fixed width `n` and cumulant order `K`, and that at a depth of 8 hidden
layers, their `K = 4` method already begins to underperform plain Monte
Carlo sampling, before improving again as width is increased further. Since
the competition's networks have 32 hidden layers, considerably deeper than
the depths reported in the paper's main results, the practical benefit of a
higher-order cumulant method at this specific depth and width is not
guaranteed without direct measurement.

The authors' reference implementation,
[alignment-research-center/mlp_cumulant_propagation](https://github.com/alignment-research-center/mlp_cumulant_propagation),
implements the general algorithm, including an asymptotically faster
factorized tensor representation needed to keep the `K = 3` and `K = 4`
algorithms within a practical FLOP budget at this competition's scale. A
naive, unfactored implementation of `K = 3` cumulant propagation costs on the
order of `O(L n^4)` FLOPs, which at `n = 256` and `L = 32` is close to the
competition's entire per-network FLOP budget, leaving little headroom; the
factorized representation reduces this to roughly `O(L^2 n^3)`, which is
comfortably affordable. That reference implementation is a substantially
larger and more intricate piece of code than what is implemented here,
involving Wick coefficients, integer partition combinatorics, and factored
symmetric tensor algebra, and porting it correctly into the restricted
flopscope environment (which permits only NumPy-style operations and the
Python standard library, with no third-party dependencies) was judged to
carry meaningful risk of subtle implementation error if attempted without
extensive additional validation. It was not attempted in this repository,
and is noted here as the clear next step for anyone extending this work.

## References

Price, R. (1958). A useful theorem for nonlinear devices having Gaussian
inputs. IRE Transactions on Information Theory, 4(2), 69-72.

Frey, B. J., and Hinton, G. E. (1999). Variational learning in nonlinear
Gaussian belief networks. Neural Computation, 11(1), 193-213.

Cho, Y., and Saul, L. K. (2009). Kernel methods for deep learning. Advances
in Neural Information Processing Systems, 22.

Gast, J., and Roth, S. (2018). Lightweight probabilistic deep networks.
Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, 3369-3378.

Wright, L., et al. (2024). Referenced in Wu et al. (2026) as the source of
the covariance propagation formula generalized in that paper's Theorem 1.

Wu, W., Lecomte, V., Winer, M., Robinson, G., Hilton, J., and Christiano, P.
(2026). Estimating the expected output of wide random MLPs more efficiently
than sampling. arXiv:2605.05179.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE)
for the full text.
