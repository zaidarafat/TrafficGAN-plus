"""Networks of Section 3.2 (Figure 1) and the flow-level MLP baselines."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class Generator(nn.Module):
    """G(z, c): [z; e(c)] -> linear -> (T x 64) -> Conv1D(64) -> Conv1D(128) -> LSTM(128) -> flatten -> linear -> tanh."""

    def __init__(self, d, T, K, n_z=100, emb_dim=64, conv_channels=(64, 128), hidden=128, **_):
        super().__init__()
        c1, c2 = conv_channels
        self.T, self.d, self.K, self.n_z, self.c1 = T, d, K, n_z, c1
        self.emb = nn.Sequential(nn.Linear(K, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU())
        self.proj = nn.Linear(n_z + emb_dim, T * c1)
        self.conv = nn.Sequential(nn.Conv1d(c1, c1, 3, 1, 1), nn.ReLU(), nn.Conv1d(c1, c2, 3, 1, 1), nn.ReLU())
        self.lstm = nn.LSTM(c2, hidden, batch_first=True)
        self.out = nn.Linear(T * hidden, T * d)

    def forward(self, z, c):
        e = self.emb(F.one_hot(c, self.K).float())
        h = self.proj(torch.cat([z, e], 1)).view(-1, self.T, self.c1).transpose(1, 2)
        h = self.conv(h).transpose(1, 2)
        h, _ = self.lstm(h)
        return torch.tanh(self.out(h.reshape(h.size(0), -1))).view(-1, self.T, self.d)


class Critic(nn.Module):
    """D(x) and q(.|x): Conv1D(64) -> Conv1D(128) -> LSTM(128) -> dropout -> mean pool -> {linear score, MLP softmax head}.

    pack > 1 implements PacGAN: m same-class windows are concatenated along the feature axis."""

    def __init__(self, d, T, K, conv_channels=(64, 128), hidden=128, dropout=0.2, leaky_slope=0.2,
                 spectral=False, pack=1, conditional="projection", **_):
        super().__init__()
        c1, c2 = conv_channels
        sn = spectral_norm if spectral else (lambda m, **k: m)
        self.pack = pack
        self.conv = nn.Sequential(sn(nn.Conv1d(d * pack, c1, 3, 1, 1)), nn.LeakyReLU(leaky_slope),
                                  sn(nn.Conv1d(c1, c2, 3, 1, 1)), nn.LeakyReLU(leaky_slope))
        self.lstm = nn.LSTM(c2, hidden, batch_first=True)
        if spectral:
            self.lstm = spectral_norm(spectral_norm(self.lstm, name="weight_ih_l0"), name="weight_hh_l0")
        self.drop = nn.Dropout(dropout)
        self.score = sn(nn.Linear(hidden, 1))
        self.aux = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, K))
        # projection conditioning (Miyato & Koyama, 2018): D(x, c) = psi(h) + e_c^T h.
        # conditional="none" reproduces the unconditional critic of package v1.
        self.conditional = conditional
        if conditional == "projection":
            self.embed = sn(nn.Embedding(K, hidden))
            if not spectral:
                nn.init.normal_(self.embed.weight, 0, 0.02)

    def packed(self, x):
        if self.pack == 1:
            return x
        B, T, d = x.shape
        return x.view(B // self.pack, self.pack, T, d).permute(0, 2, 1, 3).reshape(B // self.pack, T, d * self.pack)

    def trunk(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.drop(h).mean(1)

    def forward(self, x, c=None, already_packed=False):
        """Returns (critic score [B'], class logits [B', K]) with B' = B / pack.
        c: labels of the packed samples [B'] (required when conditional == "projection")."""
        hb = self.trunk(x if already_packed else self.packed(x))
        s = self.score(hb).squeeze(1)
        if self.conditional == "projection":
            s = s + (self.embed(c) * hb).sum(1)
        return s, self.aux(hb)


class MLPGenerator(nn.Module):
    """Flow-level conditional generator for the MLP-GAN / MLP WGAN-GP baselines."""

    def __init__(self, d, K, n_z=100, hidden=256):
        super().__init__()
        self.K, self.n_z = K, n_z
        self.net = nn.Sequential(nn.Linear(n_z + K, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d), nn.Tanh())

    def forward(self, z, c):
        return self.net(torch.cat([z, F.one_hot(c, self.K).float()], 1))


class MLPCritic(nn.Module):
    def __init__(self, d, K, hidden=256, sigmoid=False):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(d + K, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, hidden),
                                 nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, x, c):
        return self.net(torch.cat([x, F.one_hot(c, self.K).float()], 1)).squeeze(1)
