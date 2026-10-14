"""
Diagnostics for a trained Tail-GAN run: why does (or doesn't) the generator learn?

    python diag_tailgan.py --out work --mode raw --epochs 3000 [--seed 1] [--ckpt path|latest]

Prints, for one real batch and one generated batch of size batch_size:
  * per-asset scale of the increments (std, mean, kurtosis)        -> is the generator's scale right?
  * empirical VaR/ES of the training strategies, real vs fake       -> what the discriminator should see
  * discriminator output (v, e) on real and on fake input           -> is D input-sensitive at all?
  * score values s_real, s_fake and the resulting losses
  * gradient of the generator loss w.r.t. the fake PnL and w.r.t. G -> does any signal reach G?
  * finite-difference sensitivity of D to a 1 % rescaling of its input
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tailgan_5_1_synthetic as tg  # noqa: E402


def kurt(x: np.ndarray) -> float:
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def main() -> None:
    p = tg.build_parser()
    a = p.parse_args(sys.argv[1:] if sys.argv[1:] and not sys.argv[1].startswith("-") else ["eval"] + sys.argv[1:])
    cfg = tg.config_from_args(a)
    device = tg.get_device()
    torch.manual_seed(cfg.seed + 100)
    rng = np.random.default_rng(cfg.seed + 100)

    R_train, R_test, port, names = tg.load_data(cfg)
    run_dir = Path(cfg.out) / "runs" / cfg.run_name
    ckpts = sorted(run_dir.glob("ckpt_E*.pt"), key=lambda q: int(q.stem.split("_E")[1]))
    ckpt_path = (ckpts[-1] if a.ckpt == "latest" else Path(a.ckpt)) if a.ckpt else run_dir / "final.pt"
    state = torch.load(ckpt_path, map_location=device)
    G, D = tg.Generator(cfg).to(device), tg.Discriminator(cfg).to(device)
    G.load_state_dict(state["G"]); D.load_state_dict(state["D"])
    G.eval(); D.eval()
    thr = dict(np.load(run_dir / "thresholds.npz"))
    strats = tg.StrategySet(cfg, port, device, thresholds=thr)          # training strategies (mode)
    print(f"run {cfg.run_name}  ckpt {ckpt_path.name}  epoch {state.get('epoch')}  K={strats.K}  device={device}")

    n = cfg.batch_size
    idx = rng.choice(len(R_train), n, replace=False)
    real = torch.as_tensor(R_train[idx], device=device)
    z = torch.distributions.StudentT(cfg.noise_df).sample((n, cfg.latent_dim)).to(device)
    fake = G(z).detach()

    # ---- 1. scale of the increments
    print("\n[1] increments per asset (real vs fake):  std x1e3 | mean x1e4 | kurtosis")
    r, f = real.cpu().numpy(), fake.cpu().numpy()
    for i, nm in enumerate(names):
        print(f"   {str(nm):12s} std {1e3*r[:, i].std():6.3f} / {1e3*f[:, i].std():6.3f}   "
              f"mean {1e4*r[:, i].mean():+6.3f} / {1e4*f[:, i].mean():+6.3f}   "
              f"kurt {kurt(r[:, i].ravel()):5.2f} / {kurt(f[:, i].ravel()):5.2f}")
    print(f"   fake values at the clamp (|x|>=0.999): {100*np.mean(np.abs(f) >= 0.999):.2f}%")
    print(f"   lag-1 autocorr real {[round(float(np.mean([np.corrcoef(x[:-1], x[1:])[0,1] for x in r[:200, i]])),3) for i in range(5)]}")
    print(f"   lag-1 autocorr fake {[round(float(np.mean([np.corrcoef(x[:-1], x[1:])[0,1] for x in f[:200, i]])),3) for i in range(5)]}")

    # ---- 2. strategy PnL tails, real vs fake, and D's view of them
    pnl_real = strats.pnl(real)
    fake_g = G(z)                                   # with graph, for gradients below
    pnl_fake = strats.pnl(fake_g)
    v_r, e_r = tg.var_es(pnl_real.detach().cpu().numpy(), cfg.alpha)
    v_f, e_f = tg.var_es(pnl_fake.detach().cpu().numpy(), cfg.alpha)
    D_real = D(pnl_real).detach().cpu().numpy()
    D_fake = D(pnl_fake).detach().cpu().numpy()
    print(f"\n[2] alpha={cfg.alpha} VaR / ES per training strategy: empirical real | empirical fake | D(real) | D(fake)")
    show = list(range(min(strats.K, 5))) + ([5, 6] if strats.K > 6 else []) + ([strats.K - 10, strats.K - 5] if strats.K > 60 else [])
    for k in show:
        print(f"   {strats.names[k]:8s} VaR {v_r[k]:+.4f} | {v_f[k]:+.4f} | {D_real[k,0]:+.4f} | {D_fake[k,0]:+.4f}"
              f"     ES {e_r[k]:+.4f} | {e_f[k]:+.4f} | {D_real[k,1]:+.4f} | {D_fake[k,1]:+.4f}")
    re_v = np.abs(v_f - v_r) / np.abs(v_r); re_e = np.abs(e_f - e_r) / np.abs(e_r)
    print(f"   mean RE over the K training strategies: VaR {100*re_v.mean():.1f}%  ES {100*re_e.mean():.1f}%")
    print(f"   |D(real) - D(fake)| mean: {np.abs(D_real - D_fake).mean():.5f}   "
          f"(vs |emp.real - emp.fake| mean {np.abs(np.stack([v_r, e_r], 1) - np.stack([v_f, e_f], 1)).mean():.5f})")
    print(f"   D(real) vs empirical real: mean|diff| {np.abs(D_real - np.stack([v_r, e_r], 1)).mean():.5f}")

    # ---- 3. scores / losses
    s_real = tg.score_quant(D(pnl_real), pnl_real, cfg.alpha, cfg.W)
    s_fake = tg.score_quant(D(pnl_fake), pnl_real, cfg.alpha, cfg.W)
    s_opt = tg.score_quant(torch.as_tensor(np.stack([v_r, e_r], 1), dtype=torch.float32, device=device),
                           pnl_real, cfg.alpha, cfg.W)
    print(f"\n[3] scores vs real PnL:  s(D(real)) {s_real.item():.5f}   s(D(fake)) {s_fake.item():.5f}   "
          f"s(empirical real VaR/ES) {s_opt.item():.5f}  <- attainable minimum")
    print(f"    loss_D = lam*s_real - s_fake = {cfg.lam*s_real.item() - s_fake.item():+.5f}    loss_G = s_fake = {s_fake.item():.5f}")

    # ---- 4. gradient signal reaching G
    G.zero_grad(set_to_none=True)
    pnl_fake.retain_grad()
    s_fake.backward()
    g_pnl = pnl_fake.grad
    g_par = torch.sqrt(sum((q.grad ** 2).sum() for q in G.parameters() if q.grad is not None))
    print(f"\n[4] d loss_G / d fake_PnL: norm {g_pnl.norm().item():.3e}, max|.| {g_pnl.abs().max().item():.3e}, "
          f"nonzero entries {100*(g_pnl != 0).float().mean().item():.1f}%")
    print(f"    d loss_G / d G_params: norm {g_par.item():.3e}")

    # ---- 5. is D sensitive to its input at all?  (rescale the fake PnL by 1 %)
    with torch.no_grad():
        d0 = D(pnl_fake.detach()); d1 = D(pnl_fake.detach() * 1.01); d2 = D(pnl_fake.detach() * 2.0)
        print(f"\n[5] D sensitivity on fake input: |D(1.01x) - D(x)| mean {(d1 - d0).abs().mean().item():.3e}, "
              f"|D(2x) - D(x)| mean {(d2 - d0).abs().mean().item():.3e}")
        d0 = D(pnl_real); d1 = D(pnl_real * 1.01); d2 = D(pnl_real * 2.0)
        print(f"    D sensitivity on real input: |D(1.01x) - D(x)| mean {(d1 - d0).abs().mean().item():.3e}, "
              f"|D(2x) - D(x)| mean {(d2 - d0).abs().mean().item():.3e}")
        wd = [q for q in D.parameters()]
        print(f"    D weight stats: first-layer |W| mean {wd[0].abs().mean().item():.4f} (fresh init ~{1/np.sqrt(cfg.batch_size)/2:.4f}), "
              f"last-layer |W| mean {wd[-2].abs().mean().item():.4f}, last bias {wd[-1].cpu().numpy().round(4)}")


if __name__ == "__main__":
    main()
