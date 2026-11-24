"""Checks Eq. (5) and Proposition 1 numerically: g_t from the reverse cumulative sum equals the autograd
gradient of D(Phi(x1, delta)) with respect to delta_t, and ||g_t|| <= sqrt(T-t) ||grad_x D||."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from tg.models import Critic
from tg.losses import increment_gradients

torch.manual_seed(0)
B, T, d, K = 4, 10, 6, 3
D = Critic(d, T, K, dropout=0.0).double()
c = torch.randint(0, K, (B,))
x = torch.randn(B, T, d, dtype=torch.double, requires_grad=True)
a = torch.autograd.grad(D(x, c)[0].sum(), x)[0]
g = increment_gradients(a)                                   # (B, T-1, d)

x1 = x[:, :1].detach().clone().requires_grad_(True)
delta = (x[:, 1:] - x[:, :-1]).detach().clone().requires_grad_(True)
x_rebuilt = torch.cat([x1, x1 + torch.cumsum(delta, 1)], 1)  # Eq. (4)
g_auto = torch.autograd.grad(D(x_rebuilt, c)[0].sum(), delta)[0]
err = (g - g_auto).abs().max().item()
print("max |g_reverse_cumsum - g_autograd| =", err)
assert err < 1e-10

gn = g.norm(dim=2); an = a.flatten(1).norm(dim=1)
bound = torch.sqrt(torch.arange(T - 1, 0, -1, dtype=torch.double))[None] * an[:, None]
assert torch.all(gn <= bound + 1e-12)
print("Proposition 1 bound holds; Eq. (5) verified.")
