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

from collections.abc import Sequence
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
# CUSTOM LOSS FUNCTIONS
# ----------------------------------------------------------------------------
class InfoNetLoss(nn.Module):
    """Differentiable InfoNet objective: regimes are where the target is persistently predicted.
 
    Call ``loss(q, features, target)`` on ONE time-ordered chunk of T consecutive rows:
 
        q         (T, K)  soft regime assignment, rows sum to 1 (e.g. softmax of a head on an embedding)
        features  (T, D)  causal features known at t (rank-gaussianise them: the proxy below is linear)
        target    (T, H)  forward risk-reward outcomes per horizon (see ``metrics.infonet_target``)
 
    Per block of ``block`` rows (every ``stride``), regime k's weighted Pearson rho between each
    feature and each horizon gives the Gaussian information proxy ``-0.5 * log(1 - rho^2)`` (nats).
    It is calibrated by the sampling null ``z^2 / (2 * n_eff)`` so a regime with too little mass in a
    block cannot score. ``n_eff`` is the weights' effective sample size divided by ``serial`` and, per
    horizon, by that horizon's bar count: an h-bar forward target overlaps its neighbours, so a block
    holds about n / h independent outcomes, not n. Pass ``horizons`` matching the target columns
    (e.g. ``(4, 16, 64)``); without it the target is treated as non-overlapping (h = 1), which makes the
    null too low and over-credits long horizons. ``serial`` (>= 1) additionally discounts serial
    dependence of the features; the metric's surrogates handle that automatically, this proxy cannot.
 
    The top ``group_size`` features form the block's group evidence. Persistence across the blocks a
    regime occupies blends the occupancy-weighted mean (``persistence=0``: average information) with the
    occupancy-weighted soft-minimum (``persistence=1``: the weakest block decides); ``tau`` (nats) sets
    the softness.
 
        loss = -sum_k usage_k * persistent_info_k  +  dwell * TV(q)  -  balance * H(usage)
 
    ``dwell`` (total variation of q over time) rewards regimes that last; ``balance`` (entropy of usage)
    stops every row collapsing into one regime. Rows with non-finite features/target carry no weight.
    Memory is O(blocks * K * D * block). This proxy only measures monotone-linear dependence, so it can
    disagree with the metric; it consumes forward outcomes by design (ex-post objective).
    """
 
    def __init__(self, block: int = 64, stride: int | None = None, group_size: int = 1,
                 persistence: float = 0.5, tau: float = 0.02, z: float = 1.96, smooth: float = 0.01,
                 min_occupancy: float = 0.2, dwell: float = 0.1, balance: float = 0.01,
                 horizons: Sequence[int] | None = None, serial: float = 1.0, eps: float = 1e-8):
        super().__init__()
        stride = block if stride is None else stride
        if block < 8 or stride < 1 or group_size < 1:
            raise ValueError("require block >= 8, stride >= 1 and group_size >= 1")
        if not 0.0 <= persistence <= 1.0 or not 0.0 <= min_occupancy < 1.0:
            raise ValueError("persistence must be in [0, 1] and min_occupancy in [0, 1)")
        if min(tau, z, smooth) <= 0 or min(dwell, balance) < 0:
            raise ValueError("tau, z, smooth must be positive; dwell, balance must be >= 0")
        if serial < 1.0:
            raise ValueError("serial must be >= 1")
        if horizons is not None:
            horizons = tuple(int(h) for h in horizons)
            if not horizons or min(horizons) < 1:
                raise ValueError("horizons must be positive integers")
        self.block, self.stride, self.group_size = int(block), int(stride), int(group_size)
        self.persistence, self.tau, self.z, self.smooth = float(persistence), float(tau), float(z), float(smooth)
        self.min_occupancy, self.dwell, self.balance, self.eps = float(min_occupancy), float(dwell), float(balance), float(eps)
        self.horizons, self.serial = horizons, float(serial)
        self.last_stats: dict[str, float] = {}
 
    def forward(self, q, features, target):
        features = features[:, None] if features.ndim == 1 else features
        target = target[:, None] if target.ndim == 1 else target
        rows, k = q.shape
        if features.shape[0] != rows or target.shape[0] != rows:
            raise ValueError("q, features and target must have the same number of rows")
        if rows < self.block:
            raise ValueError(f"need at least block={self.block} rows, got {rows}")
        if self.group_size > features.shape[1]:
            raise ValueError("group_size exceeds the number of features")
        if self.horizons is not None and len(self.horizons) != target.shape[1]:
            raise ValueError(f"horizons has {len(self.horizons)} entries but target has {target.shape[1]} columns")
        eps = self.eps
        finite = torch.isfinite(features).all(-1) & torch.isfinite(target).all(-1)
        q_all, q = q, q * finite.unsqueeze(-1).to(q.dtype)             # unusable rows carry no weight
        x = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        w = q.unfold(0, self.block, self.stride)                       # (NB, K, L)
        xs, ys = x.unfold(0, self.block, self.stride), y.unfold(0, self.block, self.stride)   # (NB, D|H, L)
 
        mass = w.sum(-1)                                               # (NB, K)
        sw = mass.clamp_min(eps)[..., None]
        n_eff = mass * mass / (w * w).sum(-1).clamp_min(eps)           # (NB, K) effective weight count
        mx, my = torch.einsum("nkl,ndl->nkd", w, xs) / sw, torch.einsum("nkl,nhl->nkh", w, ys) / sw
        vx = (torch.einsum("nkl,ndl->nkd", w, xs * xs) / sw - mx * mx).clamp_min(eps)
        vy = (torch.einsum("nkl,nhl->nkh", w, ys * ys) / sw - my * my).clamp_min(eps)
        exy = torch.einsum("nkdl,nhl->nkdh", w[:, :, None, :] * xs[:, None, :, :], ys) / sw[..., None]
        cov = exy - mx[..., :, None] * my[..., None, :]
        rho2 = (cov * cov / (vx[..., :, None] * vy[..., None, :])).clamp(0.0, 1.0 - 1e-6)
        info = -0.5 * torch.log1p(-rho2)                               # (NB, K, D, H) nats
 
        # Sampling null per (block, regime, horizon): overlap of h-bar targets leaves ~ n_eff / h independent outcomes.
        divisor = self.serial * (torch.as_tensor(self.horizons, dtype=info.dtype, device=info.device)
                                 if self.horizons is not None
                                 else torch.ones(target.shape[1], dtype=info.dtype, device=info.device))
        n_independent = (n_eff[..., None] / divisor).clamp_min(1.0)    # (NB, K, H)
        null = (self.z ** 2 / (2.0 * n_independent))[:, :, None, :]    # (NB, K, 1, H)
        evidence = self.smooth * F.softplus((info - null) / self.smooth)   # smooth relu: keeps gradient below null
        group = evidence.topk(self.group_size, dim=2).values.mean(2)  # (NB, K, H) top-m feature group
 
        occupancy = mass / self.block
        weight = (occupancy * (occupancy >= self.min_occupancy).to(occupancy.dtype))[..., None]   # (NB, K, 1)
        total = weight.sum(0).clamp_min(eps)                           # (K, 1)
        mean_info = (weight * group).sum(0) / total                    # (K, H)
        soft_min = -self.tau * (torch.logsumexp(weight.clamp_min(1e-12).log() - group / self.tau, dim=0) - total.log())
        persistent = torch.where(weight.sum(0) > 1e-6,
                                 (1.0 - self.persistence) * mean_info + self.persistence * soft_min,
                                 torch.zeros_like(mean_info))
        score = persistent.mean(-1)                                    # (K,)
 
        usage = q.sum(0) / finite.sum().clamp_min(1)
        both = (finite[1:] & finite[:-1]).to(q.dtype)
        tv = ((q_all[1:] - q_all[:-1]).abs().sum(-1) * both).sum() / both.sum().clamp_min(1.0)
        entropy = -(usage * torch.log(usage + eps)).sum()
        main = -(usage * score).sum()
        self.last_stats = {"infonet": float(-main.detach()), "tv": float(tv.detach()), "usage_entropy": float(entropy.detach())}
        return main + self.dwell * tv - self.balance * entropy
 
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