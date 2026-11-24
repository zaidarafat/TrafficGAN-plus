"""Gradient penalty (Eq. 3), temporal gradient penalty (Eqs. 5-6), auxiliary loss (Eq. 9) and ASE (Eqs. 10-11)."""
import contextlib
import torch
import torch.nn.functional as F


def input_gradient(critic, x_tilde, c=None, disable_cudnn=True):
    """a = grad_x D(x_tilde), shape (B', T, d'), with a graph for double back-propagation."""
    x_tilde = x_tilde.detach().requires_grad_(True)
    ctx = torch.backends.cudnn.flags(enabled=False) if disable_cudnn else contextlib.nullcontext()
    with ctx:
        score, _ = critic(x_tilde, c, already_packed=True)
    (a,) = torch.autograd.grad(score.sum(), x_tilde, create_graph=True)
    return a


def gradient_penalty(a, one_sided=False):
    n = a.flatten(1).norm(dim=1)
    return (F.relu(n - 1) ** 2).mean() if one_sided else ((n - 1) ** 2).mean()


def increment_gradients(a):
    """g_t = sum_{s=t+1}^{T} a_s for t = 1..T-1 (Eq. 5): reverse cumulative sum, dropping s = 1."""
    rev = torch.flip(torch.cumsum(torch.flip(a, [1]), 1), [1])   # rev[:, s] = sum_{r >= s} a_r
    return rev[:, 1:]


def temporal_gradient_penalty(a, one_sided=False):
    n = increment_gradients(a).norm(dim=2)                         # (B', T-1)
    return (F.relu(n - 1) ** 2).mean() if one_sided else ((n - 1) ** 2).mean()


def aux_loss(logits_real, c_real, smoothing=0.1):
    return F.cross_entropy(logits_real, c_real, label_smoothing=smoothing)


def attack_scenario_entropy(logits_fake):
    """H(p_bar_B): entropy (nats) of the batch-averaged predicted class distribution."""
    p_bar = F.softmax(logits_fake, 1).mean(0)
    return -(p_bar * torch.log(p_bar + 1e-12)).sum()
