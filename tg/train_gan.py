"""Training of TrafficGAN+ and its variants (Algorithm 1), and of the flow-level MLP baselines."""
from __future__ import annotations
import contextlib, copy, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from .models import Generator, Critic, MLPGenerator, MLPCritic
from .losses import (input_gradient, gradient_penalty, temporal_gradient_penalty, aux_loss,
                     attack_scenario_entropy)
from .utils import set_seed, device, mean_ks_fast

# variant -> overrides of the train config (Table 8 and Section 4.6)
VARIANTS = {
    "trafficgan":   dict(),                                   # full model, gamma=5, beta=0.1
    "no_tgp":       dict(gamma_tgp=0.0),
    "no_ase":       dict(beta_ase=0.0),
    "wgan_gp":      dict(gamma_tgp=0.0, beta_ase=0.0),        # CNN-LSTM WGAN-GP
    "alt_sn":       dict(gamma_tgp=0.0, spectral=True),       # TGP -> spectral normalization (GP kept)
    "alt_lp":       dict(gamma_tgp=0.0, gp_one_sided=True),   # TGP -> one-sided GP (WGAN-LP)
    "alt_acgan":    dict(beta_ase=0.0, acgan_weight=1.0),     # ASE -> AC-GAN class-consistency loss
    "alt_pacgan":   dict(beta_ase=0.0, pack=2),               # ASE -> PacGAN, pack size 2
    "tgp_onesided": dict(tgp_one_sided=True),
}


class ClassSampler:
    """Draws real windows of requested classes from the training partition."""

    def __init__(self, X, y, K, rng):
        self.X, self.idx, self.rng = X, [np.where(y == k)[0] for k in range(K)], rng
        self.present = np.array([len(i) > 0 for i in self.idx])

    def labels(self, n, pack=1):
        ks = np.where(self.present)[0]
        c = self.rng.choice(ks, size=n // pack)
        return np.repeat(c, pack)

    def windows(self, c):
        return self.X[[self.rng.choice(self.idx[k]) for k in c]]


def _val_ks(G, Xv, yv, n, K, dev, rng):
    sel = rng.choice(len(Xv), size=min(n, len(Xv)), replace=False)
    real = Xv[sel]
    fake = generate(G, yv[sel], dev)
    return mean_ks_fast(real.reshape(-1, real.shape[-1]), fake.reshape(-1, fake.shape[-1]))


@torch.no_grad()
def generate(G, labels, dev, batch=2048, seed=None):
    G.eval()
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    out = []
    for i in range(0, len(labels), batch):
        c = torch.as_tensor(labels[i:i + batch], dtype=torch.long, device=dev)
        z = torch.randn(len(c), G.n_z, generator=gen).to(dev)
        out.append(G(z, c).float().cpu().numpy())
    G.train()
    return np.concatenate(out) if out else np.zeros((0,))


def train_trafficgan(data, cfg, variant="trafficgan", seed=0, out_dir=None, overrides=None, log=print):
    tc = dict(cfg["train"]); tc.update(VARIANTS[variant]); tc.update(overrides or {})
    set_seed(seed); dev = device(); rng = np.random.default_rng(seed)
    Xtr, ytr, Xva, yva = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    _, T, d = Xtr.shape
    K = int(max(ytr.max(), yva.max()) + 1)
    mc = cfg["model"]
    pack = int(tc.get("pack", 1))
    G = Generator(d, T, K, **mc).to(dev)
    D = Critic(d, T, K, spectral=tc.get("spectral", False), pack=pack, **mc).to(dev)
    oG = torch.optim.Adam(G.parameters(), lr=tc["lr"], betas=tuple(tc["betas"]))
    oD = torch.optim.Adam(D.parameters(), lr=tc["lr"], betas=tuple(tc["betas"]))
    smp = ClassSampler(Xtr, ytr, K, rng)
    B = tc["batch_size"] - tc["batch_size"] % pack
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time(); best_it = 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    nocudnn = tc["disable_cudnn_in_critic"] and dev.type == "cuda"
    for it in range(1, tc["n_iters"] + 1):
        for _ in range(tc["n_critic"]):
            c_np = smp.labels(B, pack)
            c = torch.as_tensor(c_np, device=dev)
            xr = torch.as_tensor(smp.windows(c_np), device=dev)
            with torch.no_grad():
                xf = G(torch.randn(B, G.n_z, device=dev), c)
            xr_p, xf_p = D.packed(xr), D.packed(xf)
            c_pack = c[::pack]
            eps = torch.rand(xr_p.size(0), 1, 1, device=dev)
            xt = eps * xr_p + (1 - eps) * xf_p
            ctx = torch.backends.cudnn.flags(enabled=False) if nocudnn else contextlib.nullcontext()
            with ctx:
                sr, lr_ = D(xr_p, c_pack, already_packed=True)
                sf, _ = D(xf_p, c_pack, already_packed=True)
            a = input_gradient(D, xt, c_pack, disable_cudnn=nocudnn)
            gp = gradient_penalty(a, one_sided=tc.get("gp_one_sided", False))
            loss = sf.mean() - sr.mean() + tc["lambda_gp"] * gp
            if tc["gamma_tgp"] > 0:
                loss = loss + tc["gamma_tgp"] * temporal_gradient_penalty(a, one_sided=tc["tgp_one_sided"])
            loss = loss + tc["eta_cls"] * aux_loss(lr_, c_pack, tc["label_smoothing"])
            oD.zero_grad(set_to_none=True); loss.backward(); oD.step()
        # generator step
        c_np = smp.labels(B, pack)
        c = torch.as_tensor(c_np, device=dev)
        xf = G(torch.randn(B, G.n_z, device=dev), c)
        ctx = torch.backends.cudnn.flags(enabled=False) if nocudnn else contextlib.nullcontext()
        with ctx:
            sf, lf = D(xf, c[::pack])
        lossG = -sf.mean()
        ase = attack_scenario_entropy(lf)
        if tc["beta_ase"] > 0:
            lossG = lossG - tc["beta_ase"] * ase
        if tc.get("acgan_weight", 0) > 0:
            lossG = lossG + tc["acgan_weight"] * F.cross_entropy(lf, c[::pack])
        oG.zero_grad(set_to_none=True); lossG.backward(); oG.step()
        if it % tc["eval_every"] == 0 or it == tc["n_iters"]:
            ks = _val_ks(G, Xva, yva, tc["val_samples"], K, dev, rng)
            with torch.no_grad():
                w_est = (sr.mean() - sf.mean()).item()
            hist.append(dict(iter=it, loss_D=loss.item(), loss_G=lossG.item(), ase_nats=ase.item(),
                             critic_W_estimate=w_est, val_ks=ks, elapsed_s=time.time() - t0))
            log(f"[{variant} s{seed}] it {it} LD {loss.item():.3f} LG {lossG.item():.3f} val_KS {ks:.4f}")
            if ks < best - 1e-4:
                best, best_state, bad, best_it = ks, copy.deepcopy(G.state_dict()), 0, it
            else:
                bad += 1
                if bad >= tc["patience"]:
                    log(f"  early stop at {it} (best {best:.4f} at {best_it})")
                    break
    elapsed = time.time() - t0
    G.load_state_dict(best_state)
    info = dict(variant=variant, seed=seed, iters_run=it, best_iter=best_it, best_val_ks=best,
                seconds_total=elapsed, seconds_per_iter=elapsed / it,
                peak_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None),
                config=tc, d=d, T=T, K=K)
    t1 = time.time(); _ = generate(G, np.zeros(4096, dtype=np.int64), dev, batch=1024)
    info["gen_ms_per_window"] = (time.time() - t1) / 4096 * 1000
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(G.state_dict(), os.path.join(out_dir, "generator.pt"))
        with open(os.path.join(out_dir, "train_info.json"), "w") as fh:
            json.dump(info, fh, indent=1, default=float)
        import csv
        with open(os.path.join(out_dir, "log.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(hist[0].keys())); wr.writeheader(); wr.writerows(hist)
    return G, info


def train_mlp_gan(data, cfg, wasserstein=True, seed=0, n_iters=None, out_dir=None, log=print):
    """Flow-level conditional MLP-GAN (BCE, non-saturating) or MLP WGAN-GP (lambda=10, 5 critic steps).
    Returns a callable labels -> windows that groups T generated flows per window in generation order."""
    tc = cfg["train"]; set_seed(seed); dev = device(); rng = np.random.default_rng(seed)
    Xtr, ytr = data["X_train"], data["y_train"]
    N, T, d = Xtr.shape; K = int(ytr.max() + 1)
    flows = Xtr.reshape(-1, d); fy = np.repeat(ytr, T)
    idx = [np.where(fy == k)[0] for k in range(K)]
    G, D = MLPGenerator(d, K).to(dev), MLPCritic(d, K).to(dev)
    betas = tuple(tc["betas"]) if wasserstein else (0.5, 0.999)
    oG = torch.optim.Adam(G.parameters(), lr=tc["lr"], betas=betas)
    oD = torch.optim.Adam(D.parameters(), lr=tc["lr"], betas=betas)
    B = tc["batch_size"]; n_crit = tc["n_critic"] if wasserstein else 1
    n_iters = n_iters or cfg["experiments"]["baseline_epochs"]["mlp_gan_iters"]
    t0 = time.time()
    for it in range(1, n_iters + 1):
        for _ in range(n_crit):
            c_np = rng.integers(0, K, B)
            xr = torch.as_tensor(flows[[rng.choice(idx[k]) for k in c_np]], device=dev)
            c = torch.as_tensor(c_np, device=dev)
            xf = G(torch.randn(B, G.n_z, device=dev), c).detach()
            if wasserstein:
                eps = torch.rand(B, 1, device=dev)
                xt = (eps * xr + (1 - eps) * xf).requires_grad_(True)
                (g,) = torch.autograd.grad(D(xt, c).sum(), xt, create_graph=True)
                loss = D(xf, c).mean() - D(xr, c).mean() + 10 * ((g.norm(dim=1) - 1) ** 2).mean()
            else:
                loss = F.binary_cross_entropy_with_logits(D(xr, c), torch.ones(B, device=dev)) + \
                       F.binary_cross_entropy_with_logits(D(xf, c), torch.zeros(B, device=dev))
            oD.zero_grad(); loss.backward(); oD.step()
        c = torch.as_tensor(rng.integers(0, K, B), device=dev)
        s = D(G(torch.randn(B, G.n_z, device=dev), c), c)
        lossG = -s.mean() if wasserstein else F.binary_cross_entropy_with_logits(s, torch.ones(B, device=dev))
        oG.zero_grad(); lossG.backward(); oG.step()
        if it % 1000 == 0:
            log(f"[mlp_{'wgan' if wasserstein else 'gan'} s{seed}] it {it} LD {loss.item():.3f} LG {lossG.item():.3f}")

    @torch.no_grad()
    def sampler(labels):
        G.eval()
        out = []
        for i in range(0, len(labels), 1024):
            c = torch.as_tensor(np.repeat(labels[i:i + 1024], T), device=dev)
            out.append(G(torch.randn(len(c), G.n_z, device=dev), c).cpu().numpy().reshape(-1, T, d))
        return np.concatenate(out)
    info = dict(seconds_total=time.time() - t0, iters_run=n_iters)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(G.state_dict(), os.path.join(out_dir, "generator.pt"))
        json.dump(info, open(os.path.join(out_dir, "train_info.json"), "w"), indent=1)
    return sampler, info
