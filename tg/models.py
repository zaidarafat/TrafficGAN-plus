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


# --------------------------------------------------------------------------- stronger backbone (Section 4.8)
class _SelfAttention(nn.Module):
    """Multi-head self-attention written with explicit matrix products, so that the critic can be
    differentiated twice (GP and TGP need double back-propagation; fused attention kernels do not support it)."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-1, -2) / self.dk ** 0.5, dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, self.h * self.dk)
        return self.out(y)


class _Block(nn.Module):
    """Pre-LayerNorm Transformer encoder block. LayerNorm acts per sample, so it is compatible with GP/TGP."""

    def __init__(self, d_model, n_heads, ff, dropout=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.att = _SelfAttention(d_model, n_heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.att(self.n1(x)))
        return x + self.drop(self.mlp(self.n2(x)))


class TransformerGenerator(nn.Module):
    """G(z, c) for windows of T flows. Each step receives the class embedding, a projection of the global noise,
    its own noise vector and a learned position embedding; n_layers pre-LN Transformer blocks model the
    dependencies between all flows of the window; a per-step linear head and tanh give the T x d output.
    The latent vector z holds the global part (n_z) followed by the per-step parts (T * z_step)."""

    def __init__(self, d, T, K, n_z=100, d_model=128, n_layers=3, n_heads=4, ff=256, z_step=16, **_):
        super().__init__()
        self.T, self.d, self.K, self.z_global, self.z_step = T, d, K, n_z, z_step
        self.n_z = n_z + T * z_step
        self.cls = nn.Embedding(K, d_model)
        self.zg = nn.Linear(n_z, d_model)
        self.zs = nn.Linear(z_step, d_model)
        self.pos = nn.Parameter(torch.randn(1, T, d_model) * 0.02)
        self.blocks = nn.ModuleList([_Block(d_model, n_heads, ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d)

    def forward(self, z, c):
        zg, zs = z[:, :self.z_global], z[:, self.z_global:].reshape(-1, self.T, self.z_step)
        h = self.pos + self.zs(zs) + (self.zg(zg) + self.cls(c)).unsqueeze(1)
        for b in self.blocks:
            h = b(h)
        return torch.tanh(self.out(self.norm(h)))


class TransformerCritic(nn.Module):
    """D(x, c) and q(.|x) with a Transformer trunk: linear embedding of each flow + position embedding ->
    n_layers pre-LN blocks -> LayerNorm -> mean over time. Same heads as Critic: linear score with projection
    conditioning, MLP auxiliary classifier; pack > 1 implements PacGAN."""

    def __init__(self, d, T, K, d_model=128, n_layers=3, n_heads=4, ff=256, dropout=0.1, spectral=False, pack=1,
                 conditional="projection", **_):
        super().__init__()
        sn = spectral_norm if spectral else (lambda m, **k: m)
        self.pack, self.conditional = pack, conditional
        self.inp = sn(nn.Linear(d * pack, d_model))
        self.pos = nn.Parameter(torch.randn(1, T, d_model) * 0.02)
        self.blocks = nn.ModuleList([_Block(d_model, n_heads, ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.score = sn(nn.Linear(d_model, 1))
        self.aux = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, K))
        if conditional == "projection":
            self.embed = sn(nn.Embedding(K, d_model))
            if not spectral:
                nn.init.normal_(self.embed.weight, 0, 0.02)

    packed = Critic.packed

    def trunk(self, x):
        h = self.inp(x) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.norm(h).mean(1)

    def forward(self, x, c=None, already_packed=False):
        hb = self.trunk(x if already_packed else self.packed(x))
        s = self.score(hb).squeeze(1)
        if self.conditional == "projection":
            s = s + (self.embed(c) * hb).sum(1)
        return s, self.aux(hb)
