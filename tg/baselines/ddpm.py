"""Class-conditional Gaussian denoising diffusion model for flow records, following the TabDDPM design
(Kotelnikov et al., ICML 2023): MLP denoiser with sinusoidal timestep embedding and a learned class
embedding, 1000 linear-schedule steps, epsilon prediction. Discrete columns are diffused as continuous
values and snapped to valid values afterwards (TabDDPM uses multinomial diffusion for them; this is
stated in the manuscript). Generated flows are grouped into windows of T in generation order."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import device, set_seed


def _temb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class _Denoiser(nn.Module):
    def __init__(self, d, K, width=(512, 1024, 1024, 512), dim_t=128):
        super().__init__()
        self.dim_t = dim_t
        self.cls = nn.Embedding(K, dim_t)
        self.t = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))
        self.inp = nn.Linear(d, dim_t)
        layers, prev = [], dim_t
        for w in width:
            layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(0.0)]
            prev = w
        self.mlp = nn.Sequential(*layers, nn.Linear(prev, d))

    def forward(self, x, t, c):
        return self.mlp(self.inp(x) + self.t(_temb(t, self.dim_t)) + self.cls(c))


class TabDDPM:
    def __init__(self, d, K, T_flows, steps=1000, seed=0):
        set_seed(seed); self.dev = device(); self.d, self.K, self.Tw, self.steps = d, K, T_flows, steps
        self.net = _Denoiser(d, K).to(self.dev)
        b = torch.linspace(1e-4, 0.02, steps, device=self.dev)
        self.beta, self.alpha = b, 1 - b
        self.abar = torch.cumprod(self.alpha, 0)

    def fit(self, X_flows, y_flows, iters=20000, batch=1024, lr=1e-3, log=None):
        X = torch.as_tensor(X_flows, dtype=torch.float32, device=self.dev)
        y = torch.as_tensor(y_flows, dtype=torch.long, device=self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1 - i / iters)
        for it in range(iters):
            i = torch.randint(0, len(X), (batch,), device=self.dev)
            x0, c = X[i], y[i]
            t = torch.randint(0, self.steps, (batch,), device=self.dev)
            eps = torch.randn_like(x0)
            ab = self.abar[t][:, None]
            loss = F.mse_loss(self.net(ab.sqrt() * x0 + (1 - ab).sqrt() * eps, t, c), eps)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if log and (it + 1) % 2000 == 0:
                log(f"  ddpm it {it+1} loss {loss.item():.4f}")
        return self

    @torch.no_grad()
    def sample_windows(self, labels, batch=50000):
        """labels: window labels -> (n, T, d); each window = T consecutively generated flows of that class."""
        self.net.eval()
        flow_labels = np.repeat(labels, self.Tw)
        out = []
        for s in range(0, len(flow_labels), batch):
            c = torch.as_tensor(flow_labels[s:s + batch], device=self.dev)
            x = torch.randn(len(c), self.d, device=self.dev)
            for t in reversed(range(self.steps)):
                tt = torch.full((len(c),), t, device=self.dev, dtype=torch.long)
                eps = self.net(x, tt, c)
                a, ab, b = self.alpha[t], self.abar[t], self.beta[t]
                x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
                if t > 0:
                    x = x + b.sqrt() * torch.randn_like(x)
            out.append(x.clamp(-1, 1).cpu().numpy())
        return np.concatenate(out).reshape(len(labels), self.Tw, self.d)
