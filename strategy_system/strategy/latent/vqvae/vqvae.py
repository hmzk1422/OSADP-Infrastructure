import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# ============================================================
# Reconstruction losses VQ-VAE
# ============================================================

def ae_mse(x_hat, x):
    return F.mse_loss(x_hat, x)


def ae_mae(x_hat, x):
    return F.l1_loss(x_hat, x)

# ============================================================
# VQ-VAE
#
# EMA codebook update (van den Oord et al. 2017, Appendix A.1;
# also used in Razavi et al. 2019 "VQ-VAE-2"):
#
# Instead of learning the codebook via gradient descent on
# codebook_loss = MSE(z_q, z_e.detach()), we maintain running
# averages of:
#
#   N_k = exponential moving average of "how many points are
#         currently assigned to code k"
#   m_k = exponential moving average of "sum of z_e vectors
#         currently assigned to code k"
#
# and set:
#
#   e_k = m_k / N_k
#
# This tends to be more stable than gradient descent for small
# codebooks, and removes codebook_loss from the training
# objective entirely (only reconstruction_loss + commitment
# remain gradient-based).
#
# Notable problems this addresses:
#   - Codebook Collapse: a randomly-initialized code that
#     happens to be nearest to most points early in training
#     self-reinforces, since only selected codes receive
#     updates. EMA doesn't fix this on its own, but converges
#     the SELECTED codes to genuine cluster centroids faster,
#     and Laplace smoothing below prevents near-empty codes
#     from blowing up numerically.
# ============================================================

class VQVAE(nn.Module):

    def __init__(
        self,
        n_features,
        window_size=32,
        latent_dim=8,
        n_codes=16,
        hidden_dim=128,
        beta=0.25,
        use_ema=True,
        ema_decay=0.99,
        ema_epsilon=1e-5,
    ):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.n_codes = n_codes
        self.beta = beta

        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        input_dim = (
            n_features * window_size
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # ----------------------------------------------------
        # Codebook
        # ----------------------------------------------------

        self.codebook = nn.Embedding(
            n_codes,
            latent_dim,
        )

        # No manual init here. This weight is a placeholder
        # (nn.Embedding's default N(0,1) init) and is always
        # overwritten by the k-means warm-start in
        # LatentStateModel.fit() before any real training step
        # runs, so its starting value doesn't matter.

        if self.use_ema:

            # Codebook is updated manually via EMA, not via
            # gradient descent, so it should not receive
            # gradients from the optimizer.
            self.codebook.weight.requires_grad = False

            # Running count of points assigned to each code.
            self.register_buffer(
                "ema_cluster_size",
                torch.zeros(n_codes),
            )

            # Running sum of z_e vectors assigned to each code.
            self.register_buffer(
                "ema_w",
                self.codebook.weight.data.clone(),
            )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    # ========================================================
    # Encode
    # ========================================================

    def encode(self, x):

        x = x.reshape(
            x.shape[0],
            -1,
        )

        return self.encoder(x)

    # ========================================================
    # Vector quantization
    # ========================================================

    def quantize(self, z_e):

        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_e @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(dim=1)
        )

        state_index = torch.argmin(
            distances,
            dim=1,
        )

        z_q = self.codebook(state_index)

        return (
            z_q,
            state_index,
            distances,
        )

    # ========================================================
    # EMA codebook update
    #
    # Called only during training, only when use_ema=True.
    # Wrapped in no_grad since this is a manual buffer update,
    # not part of the backprop graph.
    # ========================================================

    @torch.no_grad()
    def _ema_update(self, z_e, state_index):

        one_hot = F.one_hot(
            state_index,
            self.n_codes,
        ).type(z_e.dtype)

        # How many points landed on each code this batch.
        batch_cluster_size = one_hot.sum(dim=0)

        # Sum of z_e vectors landing on each code this batch.
        batch_sum = one_hot.T @ z_e

        self.ema_cluster_size.mul_(
            self.ema_decay
        ).add_(
            batch_cluster_size,
            alpha=1 - self.ema_decay,
        )

        self.ema_w.mul_(
            self.ema_decay
        ).add_(
            batch_sum,
            alpha=1 - self.ema_decay,
        )

        # Laplace smoothing: prevents codes with ~0 assigned
        # points from producing a division by ~0, which would
        # otherwise send that code vector to a huge, unstable
        # value the next time it's (rarely) selected.
        n = self.ema_cluster_size.sum()

        cluster_size = (
            (self.ema_cluster_size + self.ema_epsilon)
            / (n + self.n_codes * self.ema_epsilon)
            * n
        )

        new_embed = self.ema_w / cluster_size.unsqueeze(1)

        self.codebook.weight.data.copy_(new_embed)

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        z_e = self.encode(x)

        z_q, state_index, distances = (
            self.quantize(z_e)
        )

        # EMA codebook update happens here, using the batch's
        # actual (z_e, state_index) pairs, before the
        # straight-through estimator below.
        if self.use_ema and self.training:
            self._ema_update(z_e, state_index)

        # Straight-through estimator.
        z_st = (
            z_e
            + (z_q - z_e).detach()
        )

        reconstruction = self.decoder(z_st)

        x_flat = x.reshape(
            x.shape[0],
            -1,
        )

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        reconstruction_loss = F.mse_loss(
            reconstruction,
            x_flat,
        )

        # ----------------------------------------------------
        # Codebook loss
        #
        # Only used in the non-EMA (gradient-descent) codebook
        # mode. When use_ema=True, the codebook is updated
        # directly via _ema_update above, so this term is
        # zeroed out and contributes nothing to the loss or
        # gradients.
        # ----------------------------------------------------

        if self.use_ema:
            codebook_loss = torch.zeros(
                (),
                device=z_e.device,
                dtype=z_e.dtype,
            )
        else:
            codebook_loss = F.mse_loss(
                z_q,
                z_e.detach(),
            )

        # ----------------------------------------------------
        # Commitment
        #
        # Kept in BOTH modes. Even with an EMA codebook, the
        # encoder still needs a pressure pulling z_e toward
        # whichever code it was assigned, otherwise the
        # encoder can drift away from the codebook entirely.
        # ----------------------------------------------------

        commitment_loss = F.mse_loss(
            z_e,
            z_q.detach(),
        )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        loss = (
            reconstruction_loss
            + codebook_loss
            + self.beta * commitment_loss
        )

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "z_e": z_e,
            "z_q": z_q,
            "state": state_index,
            "distances": distances,
            "reconstruction": reconstruction,
        }


# ============================================================
# Training / inference wrapper
# ============================================================

class LatentStateModel:

    def __init__(
        self,
        n_features,
        window_size=32,
        latent_dim=8,
        n_codes=16,
        hidden_dim=128,
        beta=0.25,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=100,
        batch_size=256,
        device=None,
        use_ema=True,
        ema_decay=0.99,
        ema_epsilon=1e-5,
        kmeans_init=True,
        kmeans_random_state=42,
    ):

        self.kmeans_init = kmeans_init
        self.kmeans_random_state = kmeans_random_state

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.window_size = window_size
        self.n_features = n_features
        self.n_codes = n_codes

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = VQVAE(
            n_features=n_features,
            window_size=window_size,
            latent_dim=latent_dim,
            n_codes=n_codes,
            hidden_dim=hidden_dim,
            beta=beta,
            use_ema=use_ema,
            ema_decay=ema_decay,
            ema_epsilon=ema_epsilon,
        ).to(self.device)

        self.mean = None
        self.std = None

        self.fitted = False

        # ----------------------------------------------------
        # Diagnostics populated during fit().
        #
        # loss_history:
        #     one float per epoch (unchanged from before).
        #
        # usage_history:
        #     one (n_codes,) array per epoch, giving the
        #     fraction of ALL windows seen during that epoch
        #     that were assigned to each code. This is a fresh
        #     per-epoch count, independent of the EMA buffers
        #     inside the model (which track a *decayed*
        #     running estimate used only for updating the
        #     codebook itself).
        # ----------------------------------------------------

        self.loss_history = []
        self.usage_history = []

    # ========================================================
    # Normalization
    # ========================================================

    def _fit_normalizer(self, X):

        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def _normalize(self, X):
        return (X - self.mean) / self.std

    # ========================================================
    # Windows
    # ========================================================

    def _make_windows(self, X):

        if len(X) < self.window_size:
            raise ValueError(
                "Not enough observations "
                "for the requested window size."
            )

        return np.stack([
            X[i - self.window_size + 1: i + 1]
            for i in range(self.window_size - 1, len(X))
        ])

    # ========================================================
    # TRAIN
    # ========================================================

    def fit(self, X, verbose_every=None):
        """
        Train VQ-VAE.

        X shape:
            (time, features)

        verbose_every:
            if set (e.g. 10), prints loss + codebook usage
            every N epochs.

        Returns:
            self.loss_history (list of floats), for backward
            compatibility with code that does history[-1].

            Per-epoch codebook usage is also stored on
            self.usage_history — a list of (n_codes,) arrays,
            one per epoch — since usage is a distribution,
            not a single number, and doesn't fit cleanly into
            the existing loss-history return value.
        """

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (time, features)."
            )

        if X.shape[1] != self.n_features:
            raise ValueError(
                "Feature dimension does not match model."
            )

        self._fit_normalizer(X)
        X = self._normalize(X)

        windows = self._make_windows(X)

        # ----------------------------------------------------
        # Initialize codebook from actual encoder outputs
        #
        # X -> windows -> encoder -> z_e (from the untrained,
        # randomly-initialized encoder) -> k-means(z_e, k=4)
        # -> codebook.weight
        #
        # Caveat: the encoder is still randomly initialized at
        # this point, so this is a practical warm-start, not a
        # perfect initialization of the FINAL z_e distribution
        # (that would require pretraining the encoder as a
        # plain autoencoder first, then k-means-ing that).
        # ----------------------------------------------------

        if self.kmeans_init:

            with torch.no_grad():

                warmup_tensor = torch.tensor(
                    windows,
                    dtype=torch.float32,
                    device=self.device,
                )

                warmup_z_e = (
                    self.model.encode(warmup_tensor)
                    .cpu()
                    .numpy()
                )

            kmeans = KMeans(
                n_clusters=self.n_codes,
                random_state=self.kmeans_random_state,
                n_init=10,
            )

            kmeans.fit(warmup_z_e)

            centers = torch.tensor(
                kmeans.cluster_centers_,
                dtype=torch.float32,
                device=self.device,
            )

            with torch.no_grad():
                self.model.codebook.weight.copy_(centers)

            # ------------------------------------------------
            # Keep the EMA buffers consistent with the new
            # codebook, if EMA is in use.
            #
            # Without this, ema_w still holds the OLD random
            # codebook's values (it was cloned from
            # codebook.weight at construction time, before
            # k-means ran) and ema_cluster_size is still 0
            # everywhere. The first few EMA updates would then
            # partially pull the codebook back toward the
            # discarded random init before catching up to the
            # k-means solution, which defeats the point of
            # warm-starting.
            #
            # Seeding ema_w / ema_cluster_size from k-means's
            # own per-cluster sums and counts makes
            # codebook.weight = ema_w / ema_cluster_size hold
            # exactly, consistent with the EMA update rule, so
            # there's no inconsistent transient at epoch 0.
            # ------------------------------------------------

            if self.model.use_ema:

                labels = kmeans.labels_

                counts = np.bincount(
                    labels,
                    minlength=self.n_codes,
                ).astype(np.float32)

                # Avoid division by zero for any k-means
                # cluster that ended up empty (rare, but
                # possible with only 4 clusters).
                safe_counts = np.where(
                    counts == 0,
                    1.0,
                    counts,
                )

                with torch.no_grad():

                    self.model.ema_cluster_size.copy_(
                        torch.tensor(
                            counts,
                            dtype=torch.float32,
                            device=self.device,
                        )
                    )

                    # ema_w should equal center * count, so that
                    # ema_w / ema_cluster_size reproduces the
                    # k-means center exactly.
                    self.model.ema_w.copy_(
                        centers
                        * torch.tensor(
                            safe_counts,
                            dtype=torch.float32,
                            device=self.device,
                        ).unsqueeze(1)
                    )

        tensor = torch.tensor(windows, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # EMA-updated codebook has no gradient, so passing it
        # to the optimizer is harmless (it's simply skipped),
        # but filtering keeps the parameter list honest either way.
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.loss_history = []
        self.usage_history = []

        self.model.train()

        for epoch in range(self.epochs):

            total_loss = 0.0
            count = 0

            # Fresh per-epoch usage counter.
            epoch_code_counts = np.zeros(
                self.n_codes,
                dtype=np.int64,
            )

            for (batch,) in loader:

                batch = batch.to(self.device)

                output = self.model(batch)

                optimizer.zero_grad()
                output["loss"].backward()
                optimizer.step()

                n = len(batch)
                total_loss += output["loss"].item() * n
                count += n

                # Accumulate which codes this batch used.
                batch_states = (
                    output["state"].detach().cpu().numpy()
                )

                epoch_code_counts += np.bincount(
                    batch_states,
                    minlength=self.n_codes,
                )

            epoch_loss = total_loss / count
            epoch_usage = epoch_code_counts / epoch_code_counts.sum()

            self.loss_history.append(epoch_loss)
            self.usage_history.append(epoch_usage)

            if verbose_every and (
                epoch % verbose_every == 0
                or epoch == self.epochs - 1
            ):
                print(
                    f"epoch {epoch:>4} | "
                    f"loss={epoch_loss:.4f} | "
                    f"usage={np.round(epoch_usage, 3)}"
                )

        self.fitted = True

        return self.loss_history

    # ========================================================
    # Plot codebook usage over training
    # ========================================================

    def plot_usage_history(self):
        """
        One line per code, showing what fraction of windows
        were assigned to it, per epoch. Useful for seeing
        WHEN a code dies or WHEN usage stabilizes, rather
        than only inspecting the final snapshot.
        """

        import matplotlib.pyplot as plt

        usage = np.stack(self.usage_history)  # (epochs, n_codes)

        plt.figure(figsize=(10, 5))

        for code in range(self.n_codes):
            plt.plot(
                usage[:, code],
                label=f"code {code}",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Fraction of windows assigned")
        plt.title("Codebook usage over training")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ========================================================
    # LIVE / VALIDATION QUERY
    # ========================================================

    @torch.no_grad()
    def encode(self, X):

        if not self.fitted:
            raise RuntimeError("Model must be fitted first.")

        X = np.asarray(X, dtype=np.float32)
        X = self._normalize(X)

        windows = self._make_windows(X)

        tensor = torch.tensor(
            windows,
            dtype=torch.float32,
            device=self.device,
        )

        self.model.eval()

        z_e = self.model.encode(tensor)
        z_q, state, distances = self.model.quantize(z_e)

        row_index = torch.arange(len(state), device=self.device)
        quantization_distance = distances[row_index, state]

        return {
            "z_e": z_e.cpu().numpy(),
            "z_q": z_q.cpu().numpy(),
            "state": state.cpu().numpy(),
            "quantization_distance": quantization_distance.cpu().numpy(),
        }