# Fisher-NC1 vs L2: Calibration Experiments

This package validates the central claims behind the Fisher-NC1 framework
for calibration:

1. **Null-space hypothesis**: in overparameterized networks, a substantial
   fraction of within-class feature variance lies in the null space of the
   classifier `W`. This component is invisible to output distributions
   (and therefore to ECE), but L2-based NC measures count it anyway.
2. **Bound tightness**: the Fisher-based ECE bound is numerically tighter
   than the L2-based bound in this regime.
3. **Closed-form T\***: the optimal temperature `T*` predicted from the
   ETF radius and per-class accuracy matches the validation-fitted T*.

## Files

```
models.py             ResNet-18 for CIFAR with explicit feature access
nc_measures.py        Class means, within-class covariance, FNC1, NC1, ETF diagnostics,
                      and the row-space / null-space decomposition.
calibration.py        ECE, temperature scaling, theoretical T* prediction, and
                      ECE bounds (Fisher / L2 / L2-row-only).
trajectory.py         Dual-coordinate trajectory analysis: m-coords (probability
                      simplex), e-coords (centered logits), Pythagorean
                      decomposition, and ETF formation metrics.
env_setup.py          OpenMP duplicate-library workaround for Windows + Anaconda.
train.py              Train ResNet-18 on CIFAR-100 (AdamW + cosine).
train_with_snapshots.py
                      Same training loop, but snapshots class means / W / diagnostic
                      batch every N epochs for trajectory analysis.
extract_features.py   Cache (features, labels, logits, W) from a checkpoint.
exp1_null_space.py    Experiment 1: variance decomposition + per-class plots.
exp2_bound_comparison.py
                      Experiment 2: Fisher vs L2 vs L2-row-only bounds.
exp3_T_star.py        Experiment 3: theoretical T* vs validation-fitted T*.
exp4_trajectory.py    Experiment 4: m-coord / e-coord trajectories and
                      Pythagorean decomposition over training.
test_sanity.py        Synthetic-data unit tests for all utilities.
test_trajectory_sanity.py
                      Sanity tests for trajectory.py.
test_e2e_synthetic.py End-to-end pipeline test on synthetic data.
make_synthetic_cache.py     Make a synthetic feature cache (for exp1/2/3).
make_synthetic_snapshots.py Make synthetic snapshots (for exp4).
run_all.sh            One-shot end-to-end driver.
```

## Quickstart

```bash
# 0. (optional) verify the code is correct on synthetic data
python test_sanity.py

# 1. train a baseline model (~30 min on a single GPU; longer = stronger NC)
python train.py --epochs 30 --save-dir checkpoints/baseline

# 2. cache features
python extract_features.py \
    --checkpoint checkpoints/baseline/best.pt \
    --split test \
    --out cache/test_features.pt

# 3. run the three experiments
python exp1_null_space.py       --features cache/test_features.pt --out results/exp1
python exp2_bound_comparison.py --features cache/test_features.pt --out results/exp2
python exp3_T_star.py           --features cache/test_features.pt --out results/exp3
```

Or simply: `bash run_all.sh`.

## What each experiment produces

### Experiment 1: null-space hypothesis

`results/exp1/exp1_results.json` contains:
- `global.var_total`, `global.var_row`, `global.var_null`, `global.null_fraction`
- per-class breakdown
- ETF geometry diagnostics (`rho`, equinorm CV, equiangle CV)

Figures:
- `var_decomposition.pdf` — bar chart of row-space vs null-space variance.
- `eigenvalue_spectra.pdf` — log-scale eigenvalues of `P_row Sigma_W P_row` and `P_null Sigma_W P_null`.
- `per_class_null_fraction.pdf` — distribution of null-fraction across classes.

**Expected finding**: in CIFAR-100 + ResNet-18 (d=512, K=100, null dim=412),
a non-trivial fraction (often >30%) of within-class variance lives in
the null space of `W`. This is what the L2 measure conflates with
calibration-relevant spread.

### Experiment 2: bound comparison

`results/exp2/exp2_results.json` contains the three bounds (Fisher, L2-row-only, L2-total) along with empirical ECE and the bias estimate.

`results/exp2/bound_comparison.pdf` visualizes the four numbers on a log scale.

**Expected finding**: at the model's training temperature `T=1`, the Fisher
bound is closest to the empirical ECE among the three. The L2-total bound
is loosest (it includes null-space variance that does not affect ECE).

### Experiment 3: T* prediction

`results/exp3/exp3_results.json` contains:
- `T_star_theory` (closed-form prediction)
- `T_star_empirical` (grid-searched argmin of empirical ECE)
- `T_star_bound_min` (grid-searched argmin of the Fisher bound)
- `etf` block

Figures:
- `T_star_curve.pdf` — empirical ECE and Fisher bound vs `T`, with both
  T* values marked.
- `T_decomposition.pdf` — bias term and variance term of the bound vs `T`,
  showing the trade-off explicitly.

**Expected finding**: `T_star_theory` falls within ~10-20% of `T_star_empirical`.
This is achieved without using a validation set — the prediction depends only
on the ETF radius `rho^2` (measured from training-set features) and the
per-class accuracy `a` (also measurable on the training set).

### Experiment 4: Trajectory in dual coordinates

This experiment is *separate* from 1-3: it requires *snapshot-style training*
(periodic dumps of class means and classifier weights), not just a single
final checkpoint. Run:

```bash
# Train with snapshots every epoch (CIFAR-10 recommended for cleanest plots)
python train_with_snapshots.py --dataset cifar10 --epochs 30 \
       --save-dir checkpoints/snap --snapshot-every 1

# Analyze trajectories
python exp4_trajectory.py --snapshots checkpoints/snap/snapshots --out results/exp4
```

`results/exp4/exp4_summary.json` contains per-epoch rows with:
- `rho`, `equinorm_cv`, `equiangle_cv`, `etf_procrustes_l2` (ETF formation)
- `kl_q_to_barycenter`, `kl_q_to_vertex` (m-coord progress)
- `kl_p_to_q`, `kl_p_to_e` (NC1 / total in distribution space)
- `pythagorean_residual_rel`, `pythagorean_residual_max`

Figures:
- `m_trajectory.pdf` — class-mean output distributions `q_c(t) = softmax(W mu_c)`
  moving from the simplex barycenter to vertices over training. PCA projection
  of the K-dimensional probability simplex.
- `e_trajectory.pdf` — centered class logits `z_c(t) = W mu_c` expanding
  outward into the simplex ETF configuration. Aligned to final ETF.
- `pythagorean.pdf` — KL(p||e) vs predicted KL(p||q) + KL(q||e) per epoch,
  with relative residual on a separate panel. Validates the dual-flat
  structure of the softmax exponential family.
- `etf_metrics.pdf` — ρ, equiangle CV, and Procrustes shape-distance to
  ideal ETF, all per epoch.

**Expected finding**: in m-coordinates, q_c(t) traces a path from the
barycenter to the c-th vertex (this is the manifestation of NC in
distribution space). In e-coordinates, the K class logits expand outward
into a simplex ETF (this is the classical NC2 picture). The two
trajectories are Legendre-dual: the m-coord movement is concentrative
(simplex shrinks toward vertices), the e-coord movement is expansive
(logits radiate outward). This is the information-geometric content of NC.

The Pythagorean residual is a quantitative test of how exactly the
dual-flat structure holds. Small residual at convergence (typically < 1%)
confirms that the bias-variance decomposition in Theorem 2 is the
information-geometric Pythagorean theorem in disguise.

## Notes / pitfalls

- **Dimension d > K is required** for the null-space hypothesis to be
  meaningful. ResNet-18 has d=512 and CIFAR-100 has K=100, so null dim = 412.
- **ETF approximation quality** matters for Experiment 3. We report
  `equinorm_cv` and `equiangle_cv` so you can check this on your trained model.
  If either is large (>0.1), the closed-form T* prediction will be loose.
- **Training duration affects NC strength**. The PDF used 30 epochs; for
  stronger NC (and tighter T* match) train for 100+ epochs.
- **Fisher matrix** is computed implicitly via KL divergence, so it is
  numerically stable. We do not form `F(mu_c)` explicitly.
- **Pinsker slack** means absolute bound values are 5-10x larger than
  empirical ECE. This is expected and discussed in the PDF (Sec. 4.5,
  "structural vs numerical tightness"). The functional form (location of
  the minimum, not absolute value) is what the experiment validates.
- **FNC1 is a property of the trained model, not of post-hoc T**.
  Post-hoc temperature scaling does NOT touch features -- it only rescales
  logits. So in Experiment 3 we measure FNC1 ONCE at training T_0 = 1
  and hold it fixed during the T sweep. The bound's T-dependence comes
  through the `1/T` factor in the variance term and through the bias term.
- **Overconfident vs underconfident regime**. The closed-form T* derivation
  assumes the model is overconfident at T_0 = 1 (i.e., p_T(y_c|mu_c) > acc_c
  at T = 1). This is the typical regime for modern DNNs and matches the
  PDF's CIFAR-100 experiment (T*_fitted = 1.824 > 1). If a model is
  underconfident, the empirical T* will be < 1 and the closed-form
  prediction may not apply (the bound minimizer and the empirical ECE
  minimizer can lie on opposite sides of T = 1). The synthetic test
  in `test_e2e_synthetic.py` deliberately constructs an underconfident
  case to expose this; on real CIFAR-100 ResNet-18 baselines, the
  overconfident assumption holds robustly.

## Reproducing the PDF's Theorem 4 (lambda*) experiment

That requires training under the FNC1 regularization
`L = L_CE + lambda * FNC1`. Add to your training loop:

```python
from nc_measures import fisher_nc1, class_means
# inside training loop, after computing logits and h:
mu_batch = class_means(h, y, K)            # per-batch class means (rough)
fnc = fisher_nc1(h, y, model.classifier.weight, mu_batch, T=1.0, K=K)["fnc1"]
loss = F.cross_entropy(logits, y) + lam * fnc
```

The per-batch mean estimate is noisy but works in practice. A more
faithful version uses moving-average class means (EMA). See PDF Sec. 5.

## Reference

Code is structured to match the notation in the PDF. Theorem references
(Theorem 1, Theorem 2, Theorem 4) are inlined as docstring comments.
