import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Reconstruction losses VQ-VAE
#
# The reconstruction loss determines WHAT information the
# encoder is rewarded for preserving.
#
# MSE:
#   Penalizes large reconstruction errors more strongly.
#   Can encourage the model toward the average when inputs
#   contain substantial noise.
#
# MAE:
#   Penalizes errors linearly.
#   Less sensitive to large outliers and may preserve a
#   different representation of noisy market windows.
# ============================================================

def ae_mse(x_hat, x):
    return F.mse_loss(x_hat, x)


def ae_mae(x_hat, x):
    return F.l1_loss(x_hat, x)

# ============================================================
# VQ-VAE
#
# Why we choose it:
# Neural networks can learn complex, nonlinear relationships and
# compress high-dimensional market observations into a compact
# representation. VQ-VAE adds a learnable codebook, allowing
# recurring market configurations to be represented by reusable
# discrete latent states.
#
# Our objective:
# Given noisy, normalized market features, learn whether recurring
# latent configurations exist and whether they can be recognized
# again when they recur later.
#
# Let lookback window be 32 bars and feature dimension be 50 features.
# 1. Input:
#    x_t = lookback window × feature dimension
#    Example: 32 bars × 50 features = 1,600 inputs.
#
# 2. Encoder:
#    z_e = f_theta(x)
#    Compress the market window into a continuous latent vector.
#
# 3. Codebook:
#    C = {e_1, ..., e_K}
#    K learnable latent prototypes stored as vectors.
#
# 4. Vector quantization:
#    k* = argmin_k ||z_e - e_k||²
#    Find the nearest prototype and assign its state index k*.
#
# 5. Quantized latent:
#    z_q = e_k*
#    Replace the continuous representation with the selected
#    codebook prototype.
#
# 6. Reconstruction:
#    x_hat = g_phi(z_q)
#    Decode the quantized latent back into the original market
#    window.
#
# 7. Training loss:
#    L = L_recon + L_codebook + beta * L_commitment
#    Reconstruction preserves information; codebook and
#    commitment losses make latent states reusable and stable.
#
# 8. Output:
#    z_e       -> continuous latent representation
#    z_q       -> selected latent prototype
#    state     -> discrete latent-state identifier
#    distance  -> distance to assigned prototype
#
# Notable problems:
#   - Codebook Collapse: Training failure in VQ where majority code vectors
#   in dictionary are unused or dead because they don't receive updates
#   - Temporal Flickering: Responding to noise within state than underlying
#   state (HMM 105, VQ 118), [state1->state2->state1]
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
    ):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.n_codes = n_codes
        self.beta = beta

        input_dim = (
            n_features * window_size
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                latent_dim,
            ),
        )

        # ----------------------------------------------------
        # Codebook
        # ----------------------------------------------------

        self.codebook = nn.Embedding(
            n_codes,
            latent_dim,
        )

        nn.init.uniform_(
            self.codebook.weight,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                input_dim,
            ),
        )

    # ========================================================
    # Encode
    # ========================================================

    def encode(self, x):

        # x:
        # (batch, window, features)

        x = x.reshape(
            x.shape[0],
            -1,
        )

        return self.encoder(x)

    # ========================================================
    # Vector quantization
    # ========================================================

    def quantize(self, z_e):

        # z_e:
        # (batch, latent_dim)

        # codebook:
        # (n_codes, latent_dim)

        distances = (
            z_e.pow(2).sum(
                dim=1,
                keepdim=True,
            )
            - 2
            * z_e
            @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(
                dim=1
            )
        )

        state_index = torch.argmin(
            distances,
            dim=1,
        )

        z_q = self.codebook(
            state_index
        )

        return (
            z_q,
            state_index,
            distances,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        z_e = self.encode(x)

        z_q, state_index, distances = (
            self.quantize(z_e)
        )

        # Straight-through estimator.
        #
        # Forward:
        #   use quantized z_q
        #
        # Backward:
        #   gradients flow through z_e

        z_st = (
            z_e
            + (z_q - z_e).detach()
        )

        reconstruction = self.decoder(
            z_st
        )

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
        # Codebook
        # ----------------------------------------------------

        codebook_loss = F.mse_loss(
            z_q,
            z_e.detach(),
        )

        # ----------------------------------------------------
        # Commitment
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
            "reconstruction_loss":
                reconstruction_loss,
            "codebook_loss":
                codebook_loss,
            "commitment_loss":
                commitment_loss,
            "z_e":
                z_e,
            "z_q":
                z_q,
            "state":
                state_index,
            "distances":
                distances,
            "reconstruction":
                reconstruction,
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
    ):

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
        ).to(self.device)

        # Fitted normalization parameters.
        self.mean = None
        self.std = None

        self.fitted = False

    # ========================================================
    # Normalization
    # ========================================================

    def _fit_normalizer(self, X):

        self.mean = X.mean(
            axis=0,
            keepdims=True,
        )

        self.std = X.std(
            axis=0,
            keepdims=True,
        )

        self.std = np.where(
            self.std < 1e-8,
            1.0,
            self.std,
        )

    def _normalize(self, X):

        return (
            X - self.mean
        ) / self.std

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
            X[
                i - self.window_size + 1
                : i + 1
            ]
            for i in range(
                self.window_size - 1,
                len(X),
            )
        ])

    # ========================================================
    # TRAIN
    # ========================================================

    def fit(self, X):

        """
        Train VQ-VAE.

        X shape:
            (time, features)

        For synthetic EDA:
            use the full synthetic dataset.

        For real walk-forward:
            pass ONLY the historical training segment.
        """

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must have shape "
                "(time, features)."
            )

        if X.shape[1] != self.n_features:
            raise ValueError(
                "Feature dimension does not "
                "match model."
            )

        # Fit normalization on supplied data.
        self._fit_normalizer(X)

        X = self._normalize(X)

        windows = self._make_windows(X)

        tensor = torch.tensor(
            windows,
            dtype=torch.float32,
        )

        dataset = torch.utils.data.TensorDataset(
            tensor
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        history = []

        self.model.train()

        for epoch in range(
            self.epochs
        ):

            total_loss = 0.0
            count = 0

            for (batch,) in loader:

                batch = batch.to(
                    self.device
                )

                output = self.model(
                    batch
                )

                optimizer.zero_grad()

                output["loss"].backward()

                optimizer.step()

                n = len(batch)

                total_loss += (
                    output["loss"].item()
                    * n
                )

                count += n

            epoch_loss = (
                total_loss / count
            )

            history.append(
                epoch_loss
            )

        self.fitted = True

        return history

    # ========================================================
    # LIVE / VALIDATION QUERY
    # ========================================================

    @torch.no_grad()
    def encode(self, X):

        """
        Encode new data using the frozen model.

        Returns:
            z_e
            z_q
            state
            quantization_distance
        """

        if not self.fitted:
            raise RuntimeError(
                "Model must be fitted first."
            )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        X = self._normalize(X)

        windows = self._make_windows(X)

        tensor = torch.tensor(
            windows,
            dtype=torch.float32,
            device=self.device,
        )

        self.model.eval()

        z_e = self.model.encode(
            tensor
        )

        z_q, state, distances = (
            self.model.quantize(z_e)
        )

        row_index = torch.arange(
            len(state),
            device=self.device,
        )

        quantization_distance = (
            distances[
                row_index,
                state,
            ]
        )

        return {
            "z_e":
                z_e.cpu().numpy(),

            "z_q":
                z_q.cpu().numpy(),

            "state":
                state.cpu().numpy(),

            "quantization_distance":
                quantization_distance.cpu().numpy(),
        }