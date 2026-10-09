"""
encoder_networks.py
===================================================================================

Plain ``torch.nn.Module`` architectures for :class:`TorchRegimeEncoder`.

Every network implements the same two-method contract, so any of them (or your
own) can be dropped into the encoder:

    net.encode(x)  -> (batch, W)              deterministic embedding, eval path.
                                              W == embedding_dim for every net EXCEPT
                                              VQVAENet with pooling="histogram" (num_codes)
                                              or "mean_std" (2*embedding_dim); the encoder
                                              wrapper (TorchRegimeEncoder) probes the true
                                              width, so a pooling that changes W is handled.
    net.loss(x)    -> scalar tensor            training objective (no labels!)

``x`` is always ``(batch, window, features)``, already causally z-scored.
Nothing here knows about regimes, outcomes, timestamps, or pandas.

To add a model: write an ``nn.Module`` with ``encode`` and ``loss`` and add one
``elif`` to :func:`build_network`.  That is the only place that needs to change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------------------------------------------------------
# VQ-VAE
# ----------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    """Nearest-codebook lookup with straight-through gradients.

    Anti-collapse: the codebook is seeded from real encoder outputs on the first
    training batch, and codes whose smoothed usage falls below ``dead_threshold``
    are re-seeded from random encoder outputs (standard VQ-VAE dead-code restart).
    """

    def __init__(self, num_codes: int, dim: int, commitment: float = 0.25, dead_threshold: float = 0.01,
                 usage_decay: float = 0.99):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, dim)
        self.num_codes = num_codes
        self.commitment, self.dead_threshold, self.usage_decay = commitment, dead_threshold, usage_decay
        self.register_buffer("usage", torch.ones(num_codes) / num_codes)
        self.register_buffer("initialized", torch.zeros(()))
        # cumulative dead-code restart count (diagnostics only; buffered so it checkpoints)
        self.register_buffer("dead_code_resets", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _reseed(self, flat, mask):
        pick = torch.randint(0, flat.shape[0], (int(mask.sum()),), device=flat.device)
        self.codebook.weight.data[mask] = flat[pick] + 0.01 * torch.randn_like(flat[pick])

    def forward(self, z):                                   # z: (B, T, D)
        flat = z.reshape(-1, z.shape[-1])
        if self.training and self.initialized.item() == 0:
            self._reseed(flat.detach(), torch.ones(self.num_codes, dtype=torch.bool, device=z.device))
            self.initialized.fill_(1.0)
        dist = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.T
                + self.codebook.weight.pow(2).sum(1))
        idx = dist.argmin(1)
        quantized = self.codebook(idx).view_as(z)
        vq_loss = F.mse_loss(quantized, z.detach()) + self.commitment * F.mse_loss(z, quantized.detach())
        quantized = z + (quantized - z).detach()            # straight-through
        usage = F.one_hot(idx, self.num_codes).float().mean(0)
        if self.training:
            with torch.no_grad():
                self.usage.mul_(self.usage_decay).add_((1.0 - self.usage_decay) * usage)
                dead = self.usage < self.dead_threshold / self.num_codes
                if dead.any():
                    self._reseed(flat.detach(), dead)
                    self.usage[dead] = 1.0 / self.num_codes
                    self.dead_code_resets += int(dead.sum().item())
        perplexity = torch.exp(-(usage * torch.log(usage + 1e-10)).sum())
        # straight-through was already applied above (quantized = z + (quantized-z).detach());
        # do NOT repeat it here.
        return quantized, idx, vq_loss, perplexity


class VQVAENet(nn.Module):
    """Conv VQ-VAE over a trailing window.

    window -> strided conv encoder -> per-step latents -> vector quantizer
           -> transposed-conv decoder -> reconstruction of the window.
    Embedding = mean of the *quantized* latents.  NOTE: averaging quantized vectors
    yields a *blend* of codebook entries, not a single codebook vector -- the earlier
    docstring's claim that it "lives on the learned codebook" was an overstatement.
    ``pooling`` may be ``"mean"`` (default, unchanged), ``"histogram"`` (per-window
    code-usage histogram, ``num_codes`` dims), or ``"mean_std"`` (mean+std concat).
    ``last_stats`` exposes codebook perplexity: if it is ~1 the codebook has collapsed
    and the embedding is useless -- check it after fitting.  During training
    ``epoch_stats`` accumulates dataset-level perplexity / code usage / dead-code resets
    (the previous code kept only the final batch's perplexity).
    """

    def __init__(self, n_features: int, window: int, embedding_dim: int = 16,
                 num_codes: int = 64, hidden: int = 64, commitment: float = 0.25,
                 pooling: str = "mean", usage_decay: float = 0.99,
                 dead_threshold: float = 0.01):
        super().__init__()
        self.window = window
        if pooling not in ("mean", "histogram", "mean_std"):
            raise ValueError(f"unknown pooling {pooling!r}; expected 'mean', 'histogram', 'mean_std'")
        self.pooling = pooling
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(hidden, embedding_dim, 3, padding=1))
        self.latent_norm = nn.LayerNorm(embedding_dim)
        self.quantizer = VectorQuantizer(num_codes, embedding_dim, commitment,
                                         dead_threshold=dead_threshold, usage_decay=usage_decay)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, hidden, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(hidden, n_features, 3, padding=1))
        self.last_stats: dict[str, float] = {}
        # dataset-level (not last-batch) codebook diagnostics, accumulated during training
        self.epoch_stats: dict[str, float] = {}
        self._acc_counts = None            # (num_codes,) summed one-hot counts over the epoch
        self._acc_perplexity = 0.0
        self._acc_batches = 0

    def reset_epoch_stats(self):
        """Call at the start of each training epoch to reset the accumulators."""
        self.epoch_stats = {}
        self._acc_counts = None
        self._acc_perplexity = 0.0
        self._acc_batches = 0

    def _accumulate(self, idx, perplexity):
        counts = torch.bincount(idx, minlength=self.quantizer.num_codes).float()
        self._acc_counts = counts if self._acc_counts is None else self._acc_counts + counts
        self._acc_perplexity += float(perplexity.detach())
        self._acc_batches += 1
        probs = self._acc_counts / self._acc_counts.sum().clamp_min(1.0)
        used = int((self._acc_counts > 0).sum().item())
        self.epoch_stats = {
            "codes_used": used,
            "codes_total": self.quantizer.num_codes,
            "epoch_perplexity": float(torch.exp(-(probs * torch.log(probs + 1e-10)).sum())),
            "mean_batch_perplexity": self._acc_perplexity / max(self._acc_batches, 1),
            # LIFETIME cumulative counter (NOT per-epoch). Unambiguous name; the training
            # loop logs the per-epoch delta separately as ``dead_code_resets_epoch``.
            "dead_code_resets_total": int(self.quantizer.dead_code_resets.item()),
            "batches": self._acc_batches,
        }

    def _latents(self, x):
        return self.latent_norm(self.encoder(x.transpose(1, 2)).transpose(1, 2))      # (B, T', D)

    def encode(self, x):
        z = self._latents(x)
        quantized, idx, _, _ = self.quantizer(z)
        if self.pooling == "mean":
            return quantized.mean(dim=1)
        if self.pooling == "histogram":
            hist = F.one_hot(idx.reshape(x.shape[0], -1), self.quantizer.num_codes).float()
            return hist.mean(dim=1)                                                  # (B, num_codes)
        # mean_std: mean and per-dim std of the quantized vectors, concatenated.
        # correction=0 -> population std, so a single-timestep window can't yield NaN.
        return torch.cat([quantized.mean(dim=1), quantized.std(dim=1, correction=0)], dim=-1)

    def loss(self, x):
        quantized, idx, vq_loss, perplexity = self.quantizer(self._latents(x))
        recon = self.decoder(quantized.transpose(1, 2))
        recon = F.interpolate(recon, size=self.window, mode="linear", align_corners=False).transpose(1, 2)
        recon_loss = F.mse_loss(recon, x)
        # Logging: detach ONLY for the scalar cast -- the returned graph is untouched.
        self.last_stats = {"recon": float(recon_loss.detach()), "perplexity": float(perplexity.detach())}
        if self.training:
            self._accumulate(idx, perplexity)
        return recon_loss + vq_loss


# ----------------------------------------------------------------------------
# TS2Vec (contrastive)
# ----------------------------------------------------------------------------
def _instance_contrastive(z1, z2):
    batch = z1.size(0)
    if batch < 2:
        return z1.new_zeros(())
    z = torch.cat([z1, z2], 0).transpose(0, 1)                      # (T, 2B, C)
    sim = z @ z.transpose(1, 2)
    logits = torch.tril(sim, -1)[:, :, :-1] + torch.triu(sim, 1)[:, :, 1:]
    logits = -F.log_softmax(logits, -1)
    i = torch.arange(batch, device=z.device)
    return (logits[:, i, batch + i - 1].mean() + logits[:, batch + i, i].mean()) / 2


def _temporal_contrastive(z1, z2):
    steps = z1.size(1)
    if steps < 2:
        return z1.new_zeros(())
    z = torch.cat([z1, z2], 1)                                      # (B, 2T, C)
    sim = z @ z.transpose(1, 2)
    logits = torch.tril(sim, -1)[:, :, :-1] + torch.triu(sim, 1)[:, :, 1:]
    logits = -F.log_softmax(logits, -1)
    t = torch.arange(steps, device=z.device)
    return (logits[:, t, steps + t - 1].mean() + logits[:, steps + t, t].mean()) / 2


class TS2VecNet(nn.Module):
    """TS2Vec-style encoder: dilated causal convs + instance/temporal contrastive loss.

    Two overlapping random crops of each window are encoded (with random timestep
    masking); representations of the shared timestamps are pulled together.
    Simplification vs. the paper: one crop per batch and a single (non-hierarchical)
    loss scale.  Embedding = max-pool over time of the full-window representation.
    """

    def __init__(self, n_features: int, window: int, embedding_dim: int = 16,
                 hidden: int = 64, depth: int = 3, mask_prob: float = 0.5):
        super().__init__()
        self.window, self.mask_prob = window, mask_prob
        self.input = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, dilation=2 ** i) for i in range(depth)])
        self.dilations = [2 ** i for i in range(depth)]
        self.project = nn.Linear(hidden, embedding_dim)

    def _represent(self, x, mask: bool):
        h = self.input(x)
        if mask:
            keep = torch.rand(h.shape[:2], device=h.device) > self.mask_prob
            h = h * keep.unsqueeze(-1)
        h = h.transpose(1, 2)
        for conv, dilation in zip(self.blocks, self.dilations):
            h = h + F.gelu(conv(F.pad(h, (2 * dilation, 0))))        # causal, residual
        return self.project(h.transpose(1, 2))                       # (B, T, D)

    def encode(self, x):
        return self._represent(x, mask=False).max(dim=1).values

    def loss(self, x):
        length = x.size(1)
        overlap = int(torch.randint(2, length + 1, (1,)))
        left = int(torch.randint(0, length - overlap + 1, (1,)))
        right = left + overlap
        e_left = int(torch.randint(0, left + 1, (1,)))
        e_right = int(torch.randint(0, length - right + 1, (1,))) + right
        out1 = self._represent(x[:, e_left:right], mask=True)[:, -overlap:]
        out2 = self._represent(x[:, left:e_right], mask=True)[:, :overlap]
        return _instance_contrastive(out1, out2) + _temporal_contrastive(out1, out2)


# ----------------------------------------------------------------------------
# TCN / Transformer reconstruction baselines (kept from the original encoder file)
# ----------------------------------------------------------------------------
class TCNNet(nn.Module):
    def __init__(self, n_features: int, window: int, embedding_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.window = window
        self.input = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, dilation=d) for d in (1, 2, 4)])
        self.dilations = (1, 2, 4)
        self.project = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, embedding_dim))
        self.decode = nn.Linear(embedding_dim, window * n_features)
        self.n_features = n_features

    def encode(self, x):
        h = self.input(x).transpose(1, 2)
        for conv, d in zip(self.blocks, self.dilations):
            h = F.gelu(conv(F.pad(h, (2 * d, 0))))
        return self.project(h.transpose(1, 2).mean(dim=1))

    def loss(self, x):
        return F.mse_loss(self.decode(self.encode(x)).view(-1, self.window, self.n_features), x)


class TransformerNet(nn.Module):
    def __init__(self, n_features: int, window: int, embedding_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.window, self.n_features = window, n_features
        self.input = nn.Linear(n_features, hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, batch_first=True, norm_first=True)
        self.sequence = nn.TransformerEncoder(layer, num_layers=2)
        self.position = nn.Parameter(torch.zeros(1, window, hidden))
        nn.init.normal_(self.position, std=0.02)
        self.project = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, embedding_dim))
        self.decode = nn.Linear(embedding_dim, window * n_features)

    def encode(self, x):
        return self.project(self.sequence(self.input(x) + self.position).mean(dim=1))

    def loss(self, x):
        return F.mse_loss(self.decode(self.encode(x)).view(-1, self.window, self.n_features), x)


# ----------------------------------------------------------------------------
# The one place models are chosen
# ----------------------------------------------------------------------------
def build_network(model: str, n_features: int, window: int, embedding_dim: int, **params) -> nn.Module:
    """Explicit model switch.  ``params`` are that model's own hyperparameters."""
    if model == "vq-vae":
        return VQVAENet(n_features, window, embedding_dim,
                        num_codes=params.pop("num_codes", 64), hidden=params.pop("hidden", 64),
                        commitment=params.pop("commitment", 0.25), **params)
    elif model == "ts2vec":
        return TS2VecNet(n_features, window, embedding_dim,
                         hidden=params.pop("hidden", 64), depth=params.pop("depth", 3),
                         mask_prob=params.pop("mask_prob", 0.5), **params)
    elif model == "tcn":
        return TCNNet(n_features, window, embedding_dim, hidden=params.pop("hidden", 64), **params)
    elif model == "transformer":
        return TransformerNet(n_features, window, embedding_dim, hidden=params.pop("hidden", 64), **params)
    raise ValueError(f"unknown torch model {model!r}; expected 'vq-vae', 'ts2vec', 'tcn', 'transformer'")