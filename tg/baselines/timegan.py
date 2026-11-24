"""PyTorch re-implementation of TimeGAN (Yoon, Jarrett & van der Schaar, NeurIPS 2019).

Unconditional; one model is trained per class (see run_experiments.py). Data are in [-1, 1], so the
recovery network uses tanh instead of sigmoid. Three phases as in the reference implementation:
(1) embedding/recovery autoencoder, (2) supervised next-step loss, (3) joint adversarial training."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import device, set_seed


class _RNN(nn.Module):
    def __init__(self, i, h, o, layers=3, act=None):
        super().__init__()
        self.rnn = nn.GRU(i, h, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(h, o)
        self.act = act

    def forward(self, x):
        y = self.fc(self.rnn(x)[0])
        return self.act(y) if self.act else y


class TimeGAN:
    def __init__(self, d, T, hidden=64, layers=3, z_dim=None, seed=0):
        set_seed(seed); self.dev = device(); self.d, self.T = d, T
        self.z_dim = z_dim or d
        sig = torch.sigmoid
        self.E = _RNN(d, hidden, hidden, layers, sig).to(self.dev)
        self.R = _RNN(hidden, hidden, d, layers, torch.tanh).to(self.dev)
        self.G = _RNN(self.z_dim, hidden, hidden, layers, sig).to(self.dev)
        self.S = _RNN(hidden, hidden, hidden, max(layers - 1, 1), sig).to(self.dev)
        self.D = _RNN(hidden, hidden, 1, layers).to(self.dev)

    def _z(self, n):
        return torch.rand(n, self.T, self.z_dim, device=self.dev)

    def fit(self, X, iters=10000, batch=128, lr=1e-3, gamma=1.0, log=None):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        n = len(X)
        bt = lambda: X[torch.randint(0, n, (min(batch, n),), device=self.dev)]
        oER = torch.optim.Adam(list(self.E.parameters()) + list(self.R.parameters()), lr)
        oGS = torch.optim.Adam(list(self.G.parameters()) + list(self.S.parameters()), lr)
        oD = torch.optim.Adam(self.D.parameters(), lr)
        oE2 = torch.optim.Adam(list(self.E.parameters()) + list(self.R.parameters()), lr)
        for it in range(iters):                                   # phase 1: autoencoder
            x = bt(); h = self.E(x)
            loss = 10 * torch.sqrt(F.mse_loss(self.R(h), x))
            oER.zero_grad(); loss.backward(); oER.step()
        for it in range(iters):                                   # phase 2: supervised
            x = bt(); h = self.E(x).detach()
            loss = F.mse_loss(self.S(h)[:, :-1], h[:, 1:])
            oGS.zero_grad(); loss.backward(); oGS.step()
        bce = nn.BCEWithLogitsLoss()
        for it in range(iters):                                   # phase 3: joint
            for _ in range(2):
                x = bt(); z = self._z(len(x))
                h = self.E(x); e_hat = self.G(z); h_hat = self.S(e_hat); h_sup = self.S(h)
                x_hat = self.R(h_hat)
                y_f, y_fe = self.D(h_hat), self.D(e_hat)
                lu = bce(y_f, torch.ones_like(y_f)) + gamma * bce(y_fe, torch.ones_like(y_fe))
                ls = F.mse_loss(h_sup[:, :-1], h[:, 1:].detach())
                lv = (torch.abs(x_hat.std(0) - x.std(0)).mean() + torch.abs(x_hat.mean(0) - x.mean(0)).mean())
                lg = lu + 100 * torch.sqrt(ls) + 100 * lv
                oGS.zero_grad(); lg.backward(); oGS.step()
                h = self.E(x); x_t = self.R(h); h_sup = self.S(h)
                le = 10 * torch.sqrt(F.mse_loss(x_t, x)) + 0.1 * F.mse_loss(h_sup[:, :-1], h[:, 1:])
                oE2.zero_grad(); le.backward(); oE2.step()
            x = bt(); z = self._z(len(x))
            with torch.no_grad():
                h = self.E(x); e_hat = self.G(z); h_hat = self.S(e_hat)
            y_r, y_f, y_fe = self.D(h), self.D(h_hat), self.D(e_hat)
            ld = bce(y_r, torch.ones_like(y_r)) + bce(y_f, torch.zeros_like(y_f)) + gamma * bce(y_fe, torch.zeros_like(y_fe))
            if ld.item() > 0.15:
                oD.zero_grad(); ld.backward(); oD.step()
            if log and (it + 1) % 1000 == 0:
                log(f"  timegan joint it {it+1} G {lg.item():.3f} D {ld.item():.3f}")
        return self

    @torch.no_grad()
    def sample(self, n):
        out = []
        for i in range(0, n, 4096):
            m = min(4096, n - i)
            out.append(self.R(self.S(self.G(self._z(m)))).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.T, self.d), np.float32)
