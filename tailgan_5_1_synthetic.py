"""
Tail-GAN, Section 5.1 "Synthetic multi-asset scenarios"
=======================================================

A standalone PyTorch re-implementation of the Section 5.1 experiment of

    Cont, Cucuringu, Xu & Zhang (2025), "Tail-GAN: Learning to Simulate Tail Risk
    Scenarios", Management Science.

It consolidates the pieces of the official repository
(https://github.com/chaozhang-ox/Tail-GAN: gen_synthetic.py, gen_static_port.py,
gen_thresholds.py, Transform.py, util.py, TailGAN.py, Evaluation.py) that are needed
for Section 5.1 into one file, organised as in the paper:

    Appendix C.1  -> synthetic data            : gen_synthetic_data()
    Section 3.1   -> benchmark strategies      : StrategySet
    Appendix C.3  -> differentiable sorting    : neural_sort()
    Table 11      -> generator / discriminator : Generator, Discriminator
    Eq. (7)       -> score function            : score_quant()
    Algorithm 1   -> training loop             : train()
    Section 4.2   -> RE(N) / SE(N) evaluation  : evaluate()

Usage (run inside the project virtualenv):

    python tailgan_5_1_synthetic.py gen-data   --out work
    python tailgan_5_1_synthetic.py train      --out work --mode full  --epochs 3000
    python tailgan_5_1_synthetic.py eval       --out work --mode full
    python tailgan_5_1_synthetic.py all        --out work                # the three steps in a row

`--mode` selects the three Tail-GAN variants compared in Table 1:
    raw    : Tail-GAN-Raw    (buy-and-hold on each single asset only)
    static : Tail-GAN-Static (single assets + 50 static multi-asset portfolios)
    full   : Tail-GAN        (static portfolios + mean-reversion + trend-following)

Deviations from the official code are deliberate and documented inline with "NOTE:".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Configuration (Appendix C.1 data parameters and Table 11 network configuration)
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # --- data (Appendix C.1)
    n_assets: int = 5
    n_steps: int = 100             # T: observations per trading day
    n_train: int = 50_000          # N: training scenarios
    n_test: int = 10_000           # out-of-sample scenarios
    phi1: float = 0.5              # AR(1) autocorrelation of asset 2
    phi2: float = -0.12            # AR(1) autocorrelation of asset 3
    #   NOTE: the text of Appendix C.1 says -0.15, but Table 10, Figure 6 and the official
    #   code ('AR-12') all use -0.12, which is what we use.
    nu1: int = 5                   # Student-t degrees of freedom, asset 4
    nu2: int = 10                  # Student-t degrees of freedom, asset 5
    vol_low: float = 0.3           # annualised volatility range s_i ~ U[0.3, 0.5]
    vol_high: float = 0.5
    days_per_year: int = 255       # Sigma_ij = s_i s_j rho_ij / (255 T)   (official code: 250)
    kappa_range: Tuple[float, float] = (0.08, 0.12)
    beta_range: Tuple[float, float] = (0.825, 0.875)
    gamma_range: Tuple[float, float] = (0.03, 0.07)
    rescale_paths: bool = True     # official code rescales every path to the target std
    data_seed: int = 0

    # --- benchmark strategies (Section 3.1 / Table 11)
    n_port: int = 50               # static multi-asset portfolios
    cap: float = 10.0              # capital per strategy
    window: int = 10               # look-back window of the dynamic strategies
    thresholds_pct: Tuple[float, float] = (31.0, 69.0)   # z-score percentiles for signals
    n_threshold_samples: int = 100 # training scenarios used to estimate the thresholds
    mode: str = "full"             # raw | static | full

    # --- Tail-GAN (Table 11)
    alpha: float = 0.05
    latent_dim: int = 1000
    noise_df: int = 5              # noise distribution t(5)
    batch_size: int = 1000         # N_B: PnL samples fed to the discriminator
    g_hidden: Tuple[int, ...] = (128, 256, 512, 1024)
    d_hidden: Tuple[int, ...] = (256, 128)
    lr_d: float = 1e-7
    lr_g: float = 1e-6
    adam_betas: Tuple[float, float] = (0.5, 0.999)
    lam: float = 1.0               # dual parameter lambda
    W: float = 10.0                # H1(v) = -(W/2) v^2 = -5 v^2
    project: bool = True           # project (VaR, ES) onto the constraint set W v <= e
    sort_temp: float = 0.01        # temperature of the neural sort
    epochs: int = 3000
    seed: int = 1

    # --- evaluation (Section 4.2)
    eval_n: int = 1000             # N in RE(N) / SE(N)
    eval_repeats: int = 16         # resamples used for mean / std
    eval_every: int = 100          # epochs between OOS evaluations during training
    ckpt_every: int = 100

    # --- bookkeeping
    out: str = "work"
    tag: str = ""                  # extra name for the run directory

    @property
    def run_name(self) -> str:
        base = f"tailgan_{self.mode}_a{self.alpha}_E{self.epochs}_N{self.n_train}_seed{self.seed}"
        return base + (f"_{self.tag}" if self.tag else "")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Appendix C.1: synthetic data
# --------------------------------------------------------------------------------------
def random_correlation(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random correlation matrix with non-negative entries (Appendix C.1: 'elements
    uniformly sampled from [0, 1]'). A product of a uniform factor matrix is used so the
    result is guaranteed positive semi-definite."""
    A = rng.uniform(0.0, 1.0, size=(n, n))
    C = A @ A.T + 1e-3 * np.eye(n)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def draw_data_params(cfg: Config) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.data_seed)
    n, T = cfg.n_assets, cfg.n_steps
    rho = random_correlation(n, rng)
    s = rng.uniform(cfg.vol_low, cfg.vol_high, size=n)
    std = s / math.sqrt(cfg.days_per_year * T)
    cov = np.outer(std, std) * rho
    return {
        "rho": rho,
        "std": std,
        "cov": cov,
        "phi": np.array([0.0, cfg.phi1, cfg.phi2, 0.0, 0.0]),
        "nu": np.array([0, 0, 0, cfg.nu1, cfg.nu2]),
        "kappa": rng.uniform(*cfg.kappa_range, size=n),
        "beta": rng.uniform(*cfg.beta_range, size=n),
        "gamma": rng.uniform(*cfg.gamma_range, size=n),
    }


def simulate_increments(n_samples: int, cfg: Config, params: Dict[str, np.ndarray],
                        rng: np.random.Generator) -> np.ndarray:
    """Simulate `n_samples` scenarios of shape (n_assets, n_steps) of price increments.

    Asset 1: Gaussian iid.  Assets 2-3: AR(1).  Assets 4-5: GARCH(1,1) with Student-t noise.
    As in the official code, 2T+1 steps are simulated and the last T are kept (burn-in).
    """
    n, T = cfg.n_assets, cfg.n_steps
    L = 2 * T + 1
    u = rng.multivariate_normal(np.zeros(n), params["cov"], size=(n_samples, L))  # (N, L, n)
    u = np.transpose(u, (0, 2, 1))                                                  # (N, n, L)
    x = np.zeros_like(u)

    for i in range(n):
        nu = int(params["nu"][i])
        if nu > 0:                                   # GARCH(1,1) with t(nu) noise
            # NOTE: the GARCH recursion is run on *standardised* noise (unit-variance
            # Gaussian divided by sqrt(chi2/nu)) and rescaled to the target volatility
            # afterwards. Feeding the tiny-variance u directly into sigma^2 = gamma + ...
            # (as the official code does) makes the kappa * eps^2 term negligible, i.e.
            # the series degenerates to iid t noise with constant variance.
            v = rng.chisquare(nu, size=(n_samples, L))
            eta = (u[:, i, :] / params["std"][i]) / np.sqrt(v / nu)
            g, k, b = params["gamma"][i], params["kappa"][i], params["beta"][i]
            sig2 = np.zeros((n_samples, L))
            eps = np.zeros((n_samples, L))
            for t in range(1, L):
                sig2[:, t] = g + k * eps[:, t - 1] ** 2 + b * sig2[:, t - 1]
                eps[:, t] = np.sqrt(sig2[:, t]) * eta[:, t]
            uncond_var = g / (1.0 - k - b) * nu / (nu - 2)   # stationary variance of eps
            x[:, i, :] = eps * params["std"][i] / math.sqrt(uncond_var)
        elif params["phi"][i] != 0.0:                # AR(1)
            phi = params["phi"][i]
            for t in range(1, L):
                x[:, i, t] = phi * x[:, i, t - 1] + u[:, i, t]
        else:                                        # Gaussian
            x[:, i, :] = u[:, i, :]

    x = x[:, :, -T:]                                  # keep the last T increments
    if cfg.rescale_paths:
        # NOTE: the official code rescales each path so that its realised std equals the
        # target std of the asset. Disable with --no-rescale to keep the raw dynamics.
        x = x / x.std(axis=2, keepdims=True) * params["std"][None, :, None]
    return x.astype(np.float32)


def gen_static_portfolios(n_assets: int, n_port: int, rng: np.random.Generator) -> np.ndarray:
    """Long-short static portfolios (official gen_static_port.py, 'LShort').

    Returns a (n_assets, n_port) matrix whose columns are portfolio weights with unit
    L1 norm (so a portfolio with price-1 assets has unit gross exposure)."""
    n_cols = max(n_assets ** 2, n_port)
    M = rng.standard_normal((n_assets, n_cols))
    M[rng.uniform(size=M.shape) > 0.9] = 0.0         # density 0.9 as in scipy.sparse.random
    l1 = np.abs(M).sum(axis=0, keepdims=True)
    M = np.where(l1 > 0, M / np.maximum(l1, 1e-12), 1.0 / n_assets)   # guard: no empty column
    single = (np.abs(M) == 1.0).any(axis=0)          # drop columns that are a single asset
    M = M[:, ~single]
    return M[:, :n_port].astype(np.float32)


def gen_synthetic_data(cfg: Config) -> Path:
    out = Path(cfg.out) / "data"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "synthetic_5_1.npz"
    params = draw_data_params(cfg)
    rng = np.random.default_rng(cfg.data_seed + 1)

    def gen(n: int, label: str) -> np.ndarray:
        chunks, done, t0 = [], 0, time.time()
        while done < n:
            m = min(10_000, n - done)
            chunks.append(simulate_increments(m, cfg, params, rng))
            done += m
            print(f"[gen-data] {label}: {done}/{n}  ({time.time() - t0:.0f}s)", flush=True)
        return np.concatenate(chunks)

    train = gen(cfg.n_train, "train")
    test = gen(cfg.n_test, "test")
    port = gen_static_portfolios(cfg.n_assets, cfg.n_port, np.random.default_rng(cfg.data_seed + 2))
    np.savez_compressed(path, train=train, test=test, port=port,
                        asset_names=np.array(["Gauss", f"AR({cfg.phi1})", f"AR({cfg.phi2})",
                                              f"GARCH-t({cfg.nu1})", f"GARCH-t({cfg.nu2})"]),
                        **{f"param_{k}": v for k, v in params.items()})
    print(f"[gen-data] saved {path}  train {train.shape}  test {test.shape}  port {port.shape}")
    return path


def load_data(cfg: Config):
    d = np.load(Path(cfg.out) / "data" / "synthetic_5_1.npz")
    return d["train"], d["test"], d["port"], list(d["asset_names"])


# --------------------------------------------------------------------------------------
# Section 3.1: benchmark trading strategies  Pi_k : scenario -> terminal PnL
# (port of the official Transform.py / gen_thresholds.py, device-agnostic)
# --------------------------------------------------------------------------------------
def inc2price(R: torch.Tensor) -> torch.Tensor:
    """Increments (B, A, T) -> prices (B, A, T+1) with p_0 = 1."""
    p0 = torch.ones(R.shape[0], R.shape[1], 1, dtype=R.dtype, device=R.device)
    return torch.cumsum(torch.cat([p0, R], dim=2), dim=2)


def moving_average(p: torch.Tensor, w: int) -> torch.Tensor:
    """Trailing moving average of window w along the last axis; the first w-1 entries are
    left unchanged (official `movingaverage`)."""
    out = p.clone()
    out[:, :, w - 1:] = F.avg_pool1d(p, kernel_size=w, stride=1)
    return out


def _open_positions(enter: torch.Tensor, zero_cross: torch.Tensor) -> torch.Tensor:
    """Event logic shared by the mean-reversion and trend-following strategies
    (official Position_MR / Position_TF).

    enter, zero_cross : bool (M, T) signals. A position is opened at the first `enter`
    signal following a `zero_cross` (the first column always counts as a zero-cross) and
    closed at the next zero-cross. Returns float (M, T) with 1 while a position is open."""
    M, T = enter.shape
    sig = enter.int() - zero_cross.int() + (enter & zero_cross).int()     # +1 enter, -1 clear
    flat = sig.flatten()
    nz = flat != 0
    idx = torch.arange(flat.numel(), device=flat.device)[nz]
    val = flat[nz]
    prev, cur = val[:-1], val[1:]
    entry_ts = idx[1:][(prev < 0) & (cur > 0)]
    clear_ts = idx[1:][(prev > 0) & (cur < 0)]
    entry_ts = entry_ts[entry_ts % T != 0]
    clear_ts = clear_ts[clear_ts % T != 0]
    delta = torch.zeros(flat.numel(), dtype=torch.float32, device=flat.device)
    delta[entry_ts] = 1.0
    delta[clear_ts] = -1.0
    return torch.cumsum(delta.view(M, T), dim=1)


def _zero_cross(z: torch.Tensor) -> torch.Tensor:
    zc = ((z[:, :-1] < 0) & (z[:, 1:] >= 0)) | ((z[:, :-1] > 0) & (z[:, 1:] <= 0))
    first = torch.ones(z.shape[0], 1, dtype=torch.bool, device=z.device)
    return torch.cat([first, zc], dim=1)


def _cross_up(z: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    c = (z[:, :-1] < thr) & (z[:, 1:] >= thr)
    first = torch.zeros(z.shape[0], 1, dtype=torch.bool, device=z.device)
    return torch.cat([first, c], dim=1)


def _cross_down(z: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    c = (z[:, :-1] > thr) & (z[:, 1:] <= thr)
    first = torch.zeros(z.shape[0], 1, dtype=torch.bool, device=z.device)
    return torch.cat([first, c], dim=1)


def zscore_mr(prices: torch.Tensor, window: int) -> torch.Tensor:
    """Mean-reversion signal: deviation of the price from the mean of its first
    window+1 observations (official MeanRev). Returns (B*A, T+1)."""
    B, A, _ = prices.shape
    ref = prices[:, :, :window + 1].mean(dim=2, keepdim=True)
    return ((prices - ref) / 0.01).reshape(B * A, -1)


def zscore_tf(prices: torch.Tensor, window: int) -> torch.Tensor:
    """Trend-following signal: short minus long moving average (official TrendFollow)."""
    B, A, _ = prices.shape
    z = (moving_average(prices, window) - moving_average(prices, 2 * window)) / 0.01
    return z.reshape(B * A, -1)


def _strategy_pnl(prices: torch.Tensor, z: torch.Tensor, short_enter: torch.Tensor,
                  long_enter: torch.Tensor, cap: float) -> torch.Tensor:
    B, A, _ = prices.shape
    zc = _zero_cross(z)
    pos = cap * (_open_positions(long_enter, zc) - _open_positions(short_enter, zc))
    pos[:, -1] = 0.0
    p = prices.reshape(B * A, -1)
    pnl = pos[:, :-1] * (p[:, 1:] - p[:, :-1])
    return pnl.sum(dim=1).view(B, A)


class StrategySet:
    """The K benchmark strategies of the experiment. `pnl(R)` maps a batch of increment
    scenarios (B, A, T) to terminal PnLs (B, K)."""

    def __init__(self, cfg: Config, port: np.ndarray, device: torch.device,
                 thresholds: Optional[Dict[str, np.ndarray]] = None, mode: Optional[str] = None):
        self.cfg = cfg
        self.mode = mode or cfg.mode
        self.device = device
        self.port = torch.as_tensor(port, dtype=torch.float32, device=device)   # (A, P)
        self.thresholds = thresholds or {}
        self.names: List[str] = [f"BH-{i + 1}" for i in range(cfg.n_assets)]
        if self.mode in ("static", "full"):
            self.names += [f"Port-{i + 1}" for i in range(self.port.shape[1])]
        if self.mode == "full":
            self.names += [f"MR-{i + 1}" for i in range(cfg.n_assets)]
            self.names += [f"TF-{i + 1}" for i in range(cfg.n_assets)]

    @property
    def K(self) -> int:
        return len(self.names)

    # ---- thresholds (official gen_thresholds.py): percentiles of the z-scores on
    #      training scenarios. NOTE: the official code pools all assets, so every asset
    #      gets the same thresholds; we do the same.
    def fit_thresholds(self, R_train: np.ndarray) -> Dict[str, np.ndarray]:
        n = self.cfg.n_threshold_samples
        R = torch.as_tensor(R_train[:n], dtype=torch.float32, device=self.device)
        prices = inc2price(R)
        pct = list(self.cfg.thresholds_pct)
        z_mr = zscore_mr(prices, self.cfg.window).cpu().numpy()
        z_tf = zscore_tf(prices, self.cfg.window).cpu().numpy()
        self.thresholds = {
            "MR": np.percentile(z_mr, pct).astype(np.float32),   # [31st, 69th]
            "TF": np.percentile(z_tf, pct).astype(np.float32),
        }
        return self.thresholds

    def _thr(self, key: str, j: int) -> torch.Tensor:
        return torch.as_tensor(self.thresholds[key][j], device=self.device)

    def pnl(self, R: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        prices = inc2price(R)                                       # (B, A, T+1)
        out = [cfg.cap * (prices[:, :, -1] - prices[:, :, 0])]      # buy-and-hold, (B, A)
        if self.mode in ("static", "full"):
            port_prices = torch.einsum("bat,ap->bpt", prices, self.port)
            out.append(cfg.cap * (port_prices[:, :, -1] - port_prices[:, :, 0]))
        if self.mode == "full":
            lo, hi = 0, 1
            # mean-reversion: short after z crosses up through the 69th pct, long after it
            # crosses down through the 31st pct, close on zero-crossing
            z = zscore_mr(prices, cfg.window)
            out.append(_strategy_pnl(prices, z, _cross_up(z, self._thr("MR", hi)),
                                     _cross_down(z, self._thr("MR", lo)), cfg.cap))
            # trend-following: short after z crosses down through the 31st pct, long after
            # it crosses up through the 69th pct, close on zero-crossing
            z = zscore_tf(prices, cfg.window)
            out.append(_strategy_pnl(prices, z, _cross_down(z, self._thr("TF", lo)),
                                     _cross_up(z, self._thr("TF", hi)), cfg.cap))
        return torch.cat(out, dim=1)                                # (B, K)


# --------------------------------------------------------------------------------------
# Appendix C.3: differentiable sorting (Grover et al. 2019), official util.py
# --------------------------------------------------------------------------------------
def neural_sort(s: torch.Tensor, tau: float) -> torch.Tensor:
    """s: (K, n, 1) -> relaxed permutation matrices (K, n, n) sorting in decreasing order."""
    n = s.shape[1]
    one = torch.ones(n, 1, dtype=s.dtype, device=s.device)
    A = torch.abs(s - s.transpose(1, 2))                        # (K, n, n)
    Bm = A @ (one @ one.T)                                      # row sums broadcast
    scaling = (n + 1 - 2 * (torch.arange(n, device=s.device, dtype=s.dtype) + 1))
    C = s @ scaling.unsqueeze(0)                                # (K, n, n)
    P_max = (C - Bm).transpose(1, 2)
    return F.softmax(P_max / tau, dim=-1)


# --------------------------------------------------------------------------------------
# Table 11: generator and discriminator
# --------------------------------------------------------------------------------------
class Generator(nn.Module):
    """MLP: latent (1000) -> 128 -> 256 -> 512 -> 1024 -> A*T, LeakyReLU, batch-norm."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.shape = (cfg.n_assets, cfg.n_steps)
        layers: List[nn.Module] = []
        d_in = cfg.latent_dim
        for i, h in enumerate(cfg.g_hidden):
            layers.append(nn.Linear(d_in, h))
            if i > 0:
                layers.append(nn.BatchNorm1d(h, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            d_in = h
        layers.append(nn.Linear(d_in, int(np.prod(self.shape))))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(self.net(z), -1.0, 1.0)
        return x.view(z.shape[0], *self.shape)


class Discriminator(nn.Module):
    """Takes the sorted PnL sample of each strategy (K, n) and returns (K, 2) = (VaR, ES).
    MLP: n (=1000) -> 256 -> 128 -> 2, LeakyReLU, no batch-norm."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.alpha, self.W, self.project, self.tau = cfg.alpha, cfg.W, cfg.project, cfg.sort_temp
        layers: List[nn.Module] = []
        d_in = cfg.batch_size
        for h in cfg.d_hidden:
            layers += [nn.Linear(d_in, h), nn.LeakyReLU(0.2, inplace=True)]
            d_in = h
        layers.append(nn.Linear(d_in, 2))
        self.net = nn.Sequential(*layers)

    def project_op(self, ve: torch.Tensor) -> torch.Tensor:
        """Projection onto the constraint set {W v <= e} (for alpha < 0.5) where the score
        of Eq. (7) is a proper scoring rule."""
        v, e = ve[:, 0], ve[:, 1]
        sign = 1.0 if self.alpha < 0.5 else -1.0
        inside = (self.W * v < e).float()
        v_proj = (v + self.W * e) / (1.0 + self.W ** 2)
        v_new = sign * (inside * v + (1 - inside) * v_proj)
        e_new = sign * (inside * e + (1 - inside) * self.W * v_proj)
        return torch.stack([v_new, e_new], dim=1)

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """pnl: (n, K) PnL samples -> (K, 2)."""
        s = pnl.T.unsqueeze(-1)                                 # (K, n, 1)
        P = neural_sort(s, self.tau)
        sorted_pnl = torch.bmm(P, s).squeeze(-1)                # (K, n), decreasing
        ve = self.net(sorted_pnl)
        return self.project_op(ve) if self.project else ve


# --------------------------------------------------------------------------------------
# Eq. (7): score function S_alpha(v, e, x) with H1(v) = -(W/2) v^2, H2(e) = (alpha/2) e^2
# (official S_quant). Averaged over samples and strategies: 1/(K n) sum_k sum_i.
# --------------------------------------------------------------------------------------
def score_quant(ve: torch.Tensor, X: torch.Tensor, alpha: float, W: float) -> torch.Tensor:
    """ve: (K, 2) predicted (VaR, ES);  X: (n, K) PnL samples the prediction is scored on."""
    v = ve[:, [0]]                                              # (K, 1)
    e = ve[:, [1]]
    x = X.T                                                     # (K, n)
    if alpha < 0.5:
        ind = (x <= v).float()
        G1 = lambda t: -W * t ** 2 / 2
        rt = ((ind - alpha) * (G1(v) - G1(x))
              + (1.0 / alpha) * (alpha * e) * ind * (v - x)
              + (alpha * e) * (e - v)
              - alpha * e ** 2 / 2)
    else:
        a = 1.0 - alpha
        ind = (x >= v).float()
        G1 = lambda t: -W * t ** 2 / 2
        rt = ((ind - a) * (G1(v) - G1(x))
              + (1.0 / a) * (a * (-e)) * ind * (x - v)
              + (a * (-e)) * (v - e)
              - a * e ** 2 / 2)
    return rt.mean()


# --------------------------------------------------------------------------------------
# Section 4.2: empirical VaR / ES and the relative errors RE(N), SE(N)
# --------------------------------------------------------------------------------------
def var_es(pnl: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """pnl: (n, K) -> alpha-VaR and alpha-ES per strategy (lower tail)."""
    var = np.percentile(pnl, alpha * 100, axis=0)
    mask = pnl <= var[None, :]
    es = np.where(mask, pnl, np.nan)
    es = np.nanmean(es, axis=0)
    return var, es


def relative_error(pnl_sample: np.ndarray, truth: Tuple[np.ndarray, np.ndarray],
                   alpha: float) -> Tuple[float, np.ndarray]:
    """RE of Section 4.2 for one sample: (overall, per-strategy VaR/ES errors (K, 2))."""
    v, e = var_es(pnl_sample, alpha)
    tv, te = truth
    err = np.stack([np.abs(v - tv) / np.abs(tv), np.abs(e - te) / np.abs(te)], axis=1)
    return float(err.mean()), err


@torch.no_grad()
def pnl_numpy(strats: StrategySet, R: np.ndarray, device: torch.device, chunk: int = 5000) -> np.ndarray:
    out = []
    for i in range(0, len(R), chunk):
        out.append(strats.pnl(torch.as_tensor(R[i:i + chunk], device=device)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def sample_generator(G: nn.Module, cfg: Config, n: int, device: torch.device,
                     gen: Optional[torch.Generator] = None) -> np.ndarray:
    G.eval()
    out = []
    for i in range(0, n, 5000):
        m = min(5000, n - i)
        z = torch.distributions.StudentT(cfg.noise_df).sample((m, cfg.latent_dim)).to(device)
        out.append(G(z).cpu().numpy())
    G.train()
    return np.concatenate(out)


def evaluate_samples(fake: np.ndarray, real_test: np.ndarray, strats: StrategySet, cfg: Config,
                     device: torch.device, rng: np.random.Generator) -> Dict[str, object]:
    """RE(N) of the generated scenarios and SE(N) of real samples of the same size,
    both relative to the VaR/ES computed on the whole test set."""
    pnl_real = pnl_numpy(strats, real_test, device)
    pnl_fake = pnl_numpy(strats, fake, device)
    truth = var_es(pnl_real, cfg.alpha)

    def repeated(pnl: np.ndarray):
        tot, per = [], []
        for _ in range(cfg.eval_repeats):
            idx = rng.choice(len(pnl), cfg.eval_n, replace=False)
            t, p = relative_error(pnl[idx], truth, cfg.alpha)
            tot.append(t)
            per.append(p)
        return float(np.mean(tot)), float(np.std(tot)), np.mean(per, axis=0)

    re_mean, re_std, re_per = repeated(pnl_fake)
    se_mean, se_std, se_per = repeated(pnl_real)
    return {"RE_mean": re_mean, "RE_std": re_std, "SE_mean": se_mean, "SE_std": se_std,
            "RE_per_strategy": re_per, "SE_per_strategy": se_per,
            "truth_var": truth[0], "truth_es": truth[1], "names": strats.names}


# --------------------------------------------------------------------------------------
# Algorithm 1: training
# --------------------------------------------------------------------------------------
def train(cfg: Config) -> Path:
    device = get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    R_train, R_test, port, asset_names = load_data(cfg)
    assert cfg.n_train <= len(R_train), f"only {len(R_train)} training scenarios on disk"
    R_train = R_train[:cfg.n_train]
    assert cfg.n_train % cfg.batch_size == 0, "n_train must be a multiple of batch_size"

    run_dir = Path(cfg.out) / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # strategies used for training (depends on --mode) and for evaluation (always full)
    strats = StrategySet(cfg, port, device)
    thr = strats.fit_thresholds(R_train)
    np.savez(run_dir / "thresholds.npz", **thr)
    strats_eval = StrategySet(cfg, port, device, thresholds=thr, mode="full")
    print(f"[train] device={device}  mode={cfg.mode}  K={strats.K} strategies  "
          f"thresholds MR={thr['MR']}  TF={thr['TF']}", flush=True)

    G, D = Generator(cfg).to(device), Discriminator(cfg).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.adam_betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.adam_betas)
    noise = torch.distributions.StudentT(cfg.noise_df)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tb"))
    except Exception:                                           # tensorboard not installed
        writer = None

    data = torch.as_tensor(R_train, device=device)
    n_batches = cfg.n_train // cfg.batch_size
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        perm = torch.randperm(cfg.n_train, device=device)
        loss_D_l, loss_G_l = [], []
        for b in range(n_batches):
            real = data[perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]]
            pnl_real = strats.pnl(real)                              # Pi_k(p_n), (n, K)

            # ---- discriminator: L_D = lambda * S(D(Pi(real)); Pi(real)) - S(D(Pi(G(z))); Pi(real))
            z = noise.sample((cfg.batch_size, cfg.latent_dim)).to(device)
            with torch.no_grad():
                fake = G(z)
            pnl_fake = strats.pnl(fake)
            s_real = score_quant(D(pnl_real), pnl_real, cfg.alpha, cfg.W)
            s_fake = score_quant(D(pnl_fake), pnl_real, cfg.alpha, cfg.W)
            loss_D = cfg.lam * s_real - s_fake
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            # ---- generator: L_G = S(D(Pi(G(z~))); Pi(real))
            z = noise.sample((cfg.batch_size, cfg.latent_dim)).to(device)
            pnl_fake = strats.pnl(G(z))
            loss_G = score_quant(D(pnl_fake), pnl_real.detach(), cfg.alpha, cfg.W)
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            loss_D_l.append(loss_D.item())
            loss_G_l.append(loss_G.item())

        lD, lG = float(np.mean(loss_D_l)), float(np.mean(loss_G_l))
        rec = {"epoch": epoch, "loss_D": lD, "loss_G": lG}
        if writer:
            writer.add_scalar("loss/D", lD, epoch)
            writer.add_scalar("loss/G", lG, epoch)
        msg = f"[Epoch {epoch}/{cfg.epochs}] D {lD:.5f}  G {lG:.5f}  ({time.time() - t0:.0f}s)"

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            fake = sample_generator(G, cfg, max(cfg.eval_n * 4, 4000), device)
            ev = evaluate_samples(fake, R_test, strats_eval, cfg, device, rng)
            rec.update({"RE_oos": ev["RE_mean"], "RE_oos_std": ev["RE_std"], "SE": ev["SE_mean"]})
            if writer:
                writer.add_scalar("RE/oos", ev["RE_mean"], epoch)
                writer.add_scalar("RE/sampling_error", ev["SE_mean"], epoch)
            msg += f"  RE_oos {100 * ev['RE_mean']:.2f}% (SE {100 * ev['SE_mean']:.2f}%)"
        if epoch % cfg.ckpt_every == 0 or epoch == cfg.epochs:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch,
                        "config": asdict(cfg)}, run_dir / f"ckpt_E{epoch}.pt")
        history.append(rec)
        print(msg, flush=True)

    torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": cfg.epochs,
                "config": asdict(cfg)}, run_dir / "final.pt")
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)
    np.save(run_dir / "generated_10k.npy", sample_generator(G, cfg, 10_000, device))
    if writer:
        writer.close()
    print(f"[train] done -> {run_dir}")
    return run_dir


# --------------------------------------------------------------------------------------
# Section 4.2 / Table 1: out-of-sample evaluation of a trained generator
# --------------------------------------------------------------------------------------
def evaluate(cfg: Config, ckpt: Optional[str] = None) -> Dict[str, object]:
    device = get_device()
    rng = np.random.default_rng(cfg.seed + 100)
    R_train, R_test, port, asset_names = load_data(cfg)
    run_dir = Path(cfg.out) / "runs" / cfg.run_name
    ckpt_path = Path(ckpt) if ckpt else run_dir / "final.pt"
    state = torch.load(ckpt_path, map_location=device)
    G = Generator(cfg).to(device)
    G.load_state_dict(state["G"])

    thr = dict(np.load(run_dir / "thresholds.npz"))
    strats = StrategySet(cfg, port, device, thresholds=thr, mode="full")
    fake = sample_generator(G, cfg, 10_000, device)
    ev = evaluate_samples(fake, R_test, strats, cfg, device, rng)

    print(f"\n=== Out-of-sample evaluation ({ckpt_path.name}, alpha={cfg.alpha}, N={cfg.eval_n}, "
          f"{cfg.eval_repeats} resamples) ===")
    print(f"SE({cfg.eval_n})  sampling error  : {100 * ev['SE_mean']:.2f}%  ({100 * ev['SE_std']:.2f})")
    print(f"RE({cfg.eval_n})  {cfg.run_name:16s}: {100 * ev['RE_mean']:.2f}%  ({100 * ev['RE_std']:.2f})")

    groups = {"Buy-and-hold": "BH-", "Static portfolios": "Port-", "Mean-reversion": "MR-",
              "Trend-following": "TF-"}
    names = np.array(ev["names"])
    print("\nRE by strategy group (VaR / ES, %):")
    for label, prefix in groups.items():
        m = np.char.startswith(names, prefix)
        if m.any():
            per = ev["RE_per_strategy"][m]
            print(f"  {label:18s} VaR {100 * per[:, 0].mean():6.2f}   ES {100 * per[:, 1].mean():6.2f}")

    import csv
    with open(run_dir / "eval_oos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "true_VaR", "true_ES", "RE_VaR", "RE_ES", "SE_VaR", "SE_ES"])
        for k, nm in enumerate(ev["names"]):
            w.writerow([nm, ev["truth_var"][k], ev["truth_es"][k],
                        ev["RE_per_strategy"][k, 0], ev["RE_per_strategy"][k, 1],
                        ev["SE_per_strategy"][k, 0], ev["SE_per_strategy"][k, 1]])
    summary = {k: ev[k] for k in ("RE_mean", "RE_std", "SE_mean", "SE_std")}
    (run_dir / "eval_oos.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {run_dir / 'eval_oos.csv'}")
    return ev


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["gen-data", "train", "eval", "all"])
    p.add_argument("--out", default="work", help="working directory for data, runs, logs")
    p.add_argument("--mode", choices=["raw", "static", "full"], default="full")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--n-train", type=int, default=Config.n_train)
    p.add_argument("--n-test", type=int, default=Config.n_test)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--alpha", type=float, default=Config.alpha)
    p.add_argument("--lr-d", type=float, default=Config.lr_d)
    p.add_argument("--lr-g", type=float, default=Config.lr_g)
    p.add_argument("--lam", type=float, default=Config.lam)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--data-seed", type=int, default=Config.data_seed)
    p.add_argument("--eval-every", type=int, default=Config.eval_every)
    p.add_argument("--ckpt-every", type=int, default=Config.ckpt_every)
    p.add_argument("--eval-n", type=int, default=Config.eval_n)
    p.add_argument("--eval-repeats", type=int, default=Config.eval_repeats)
    p.add_argument("--n-port", type=int, default=Config.n_port)
    p.add_argument("--no-rescale", action="store_true", help="do not rescale each path to the target std")
    p.add_argument("--no-project", action="store_true", help="disable the (VaR, ES) projection")
    p.add_argument("--ckpt", default=None, help="checkpoint to evaluate (default: final.pt of the run)")
    p.add_argument("--tag", default="", help="suffix for the run directory name")
    return p


def config_from_args(a: argparse.Namespace) -> Config:
    return Config(out=a.out, mode=a.mode, epochs=a.epochs, n_train=a.n_train, n_test=a.n_test,
                  batch_size=a.batch_size, alpha=a.alpha, lr_d=a.lr_d, lr_g=a.lr_g, lam=a.lam,
                  seed=a.seed, data_seed=a.data_seed, eval_every=a.eval_every,
                  ckpt_every=a.ckpt_every, eval_n=a.eval_n, eval_repeats=a.eval_repeats,
                  n_port=a.n_port, rescale_paths=not a.no_rescale, project=not a.no_project,
                  tag=a.tag)


def main() -> None:
    a = build_parser().parse_args()
    cfg = config_from_args(a)
    if a.command in ("gen-data", "all"):
        gen_synthetic_data(cfg)
    if a.command in ("train", "all"):
        train(cfg)
    if a.command in ("eval", "all"):
        evaluate(cfg, a.ckpt)


if __name__ == "__main__":
    main()
