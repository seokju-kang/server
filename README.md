# Tail-GAN Section 5.1 — synthetic multi-asset scenarios

`tailgan_5_1_synthetic.py` is a single-file PyTorch re-implementation of the Section 5.1
experiment of *Tail-GAN: Learning to Simulate Tail Risk Scenarios* (Cont, Cucuringu, Xu,
Zhang, Management Science 2025). It consolidates the parts of the official repository
<https://github.com/chaozhang-ox/Tail-GAN> that Section 5.1 needs, and fixes the
`your_path` placeholders, the un-formatted `{this_version}` folder names and the
`Evaluation.py` / `Screen_Ensemble` bugs of the original.

## Paper → code map

| Paper | Official file | Here |
|---|---|---|
| Appendix C.1 synthetic data (Gauss, AR(1)×2, GARCH-t(5), GARCH-t(10)) | `gen_synthetic.py` | `draw_data_params`, `simulate_increments`, `gen_synthetic_data` |
| Static long-short portfolios (50) | `gen_static_port.py` | `gen_static_portfolios` |
| Signal thresholds (31st / 69th percentile of z-scores) | `gen_thresholds.py` | `StrategySet.fit_thresholds` |
| Section 3.1 strategies: buy-and-hold, static portfolio, mean-reversion, trend-following | `Transform.py` | `StrategySet.pnl` and helpers |
| Appendix C.3 differentiable sorting | `util.py` | `neural_sort` |
| Table 11 generator / discriminator | `TailGAN.py` | `Generator`, `Discriminator` |
| Eq. (7) score with H1(v) = −5v², H2(e) = α/2·e², projection onto W·v ≤ e | `TailGAN.py` (`S_quant`, `project_op`) | `score_quant`, `Discriminator.project_op` |
| Algorithm 1 (λ = 1, lr_D = 1e-7, lr_G = 1e-6, N_B = 1000, t(5) noise) | `TailGAN.py` (`Train_Single`) | `train` |
| Section 4.2 RE(N), SE(N); Table 1 | `Evaluation.py` | `evaluate_samples`, `evaluate` |

## Usage

```bash
source .venv/bin/activate            # torch, numpy<2, scipy, tensorboard
cd ~/research/GAN/Code   # this repository, cloned on the server

# 1) data: 50,000 train + 10,000 test scenarios of 5 assets × 100 steps (≈1 min)
python tailgan_5_1_synthetic.py gen-data --out work

# 2) train Tail-GAN (Table 11 defaults: 3000 epochs). --mode raw|static|full
python -u tailgan_5_1_synthetic.py train --out work --mode full --epochs 3000

# 3) out-of-sample evaluation against the test set (Table 1 style)
python tailgan_5_1_synthetic.py eval --out work --mode full

# quick smoke test (few minutes on a GPU)
python -u tailgan_5_1_synthetic.py all --out work_smoke --n-train 5000 --n-test 2000 --epochs 20 --eval-every 10 --tag smoke
```

`eval` locates the run directory from the same flags as `train` (`--mode`, `--epochs`,
`--n-train`, `--seed`, `--alpha`, `--tag`), so pass them again, or point `--ckpt` at a
specific checkpoint (`--ckpt latest` picks the newest `ckpt_E*.pt`, useful while a run is
still training). Evaluation is seeded, so repeated calls give identical numbers.

TensorBoard logs are written to `work/runs/<run>/tb` (`tensorboard --logdir work/runs`);
checkpoints to `work/runs/<run>/ckpt_E*.pt`; evaluation to `eval_oos_<checkpoint>.csv` / `.json`.

## Raw mode (Tail-GAN-Raw) can stall: what we found and the options

With only the 5 buy-and-hold PnLs (K = 5) a run at the paper's hyper-parameters can get
stuck at ~130 % OOS error with the discriminator collapsed to a near-constant map: with
λ = 1 the discriminator objective `λ·s_real − s_fake` is zero for *any* constant D, so
nothing anchors D's output level, and at lr_D = 1e-7 a bias-dominated init never becomes
a VaR/ES estimator (its score stays ~0.27 against an attainable ~0.09). The generator then
drifts along directions D cannot see. `diag_tailgan.py` shows the signature: D(real) ≈
D(fake) ≈ 0 on the line e = W·v, and |D(2x) − D(x)| ≈ 1e-3. This is a property of the
game, not an implementation bug (the official code has the same loop; its 10-model
ensemble with selection by final generator loss would discard such a run). Two options,
both off by default:

* `--lam 10 --lr-d 1e-5`: λ > 1 makes `s_real` dominate so D is pushed to be an estimator.
  Appendix C.2 of the paper reports λ = 2 and 10 performing like λ = 1.
* `--d-warmup-steps 300`: supervised warm-start of D (minimise `s_real` only) before the
  adversarial game, then the paper's loop unchanged.

Run names include `_lam<λ>` / `_dwarm<steps>` so these runs do not collide with defaults.

## Deviations from the official code (all marked `NOTE:` in the source)

* **GARCH noise scaling.** The official code feeds the tiny-variance Gaussian `u`
  (std ≈ 0.0025) straight into `σ² = γ + κ ε² + β σ²` with γ ≈ 0.05, so `κ ε²` is
  negligible and the series is effectively iid t-noise. Here the recursion runs on
  standardised noise and is rescaled afterwards, so the ARCH effect is real.
* **φ₂ = −0.12** (paper text says −0.15; Table 10, Figure 6 and the official code use −0.12).
* **Σ uses 255 trading days** as in the paper (official code: 250).
* **Fresh noise for the generator step** (`z̃` in Algorithm 1); the official code reuses
  the discriminator-step batch with `retain_graph=True`.
* **Ground truth for RE(N)** is computed on the 10,000-scenario test set (the paper's
  out-of-sample set); the official `Evaluation.py` uses the training set.
* The discriminator loss is written as `λ·S(real) − S(fake)` and minimised, which equals
  gradient ascent on Algorithm 1's `L_D` and the official `real_score − fake_score` for λ = 1.
* Per-path rescaling to the target volatility (official behaviour) is kept by default and
  can be switched off with `--no-rescale`.
