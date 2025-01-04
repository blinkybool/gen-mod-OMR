"""
Variational Autoencoder (VAE) for MNIST digit generation, implemented with JAX.

Theory:
- VAEs learn to encode data into a lower-dimensional latent space
- The encoder learns a probabilistic mapping from data x to latent variables z
- The decoder learns to reconstruct x from z
- Training optimizes two objectives:
  1. Reconstruction quality (how well can we rebuild the input)
  2. Latent space regularity (KL divergence to a standard normal prior)
"""

import json
from typing import Tuple

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

import mattplotlib as mp


class Encoder(eqx.Module):
    """
    Neural network that learns to encode data x into approximate posterior q(z|x).
    Returns mean and log-variance of the latent Gaussian distribution.
    """

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc_mu: eqx.nn.Linear
    fc_logvar: eqx.nn.Linear

    def __init__(self, latent_dim: int, key: PRNGKeyArray):
        keys = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            key=keys[0],
        )
        self.conv2 = eqx.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            key=keys[1],
        )
        self.fc_mu = eqx.nn.Linear(64 * 7 * 7, latent_dim, key=keys[2])
        self.fc_logvar = eqx.nn.Linear(64 * 7 * 7, latent_dim, key=keys[3])

    def __call__(
        self, x: Float[Array, "1 28 28"]
    ) -> Tuple[Float[Array, " lat"], Float[Array, " lat"]]:
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = einops.rearrange(x, "c h w -> (c h w)")
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(eqx.Module):
    """
    Neural network that learns to decode latent variables z back into data space.
    Reconstructs input x through deconvolution operations.
    """

    fc: eqx.nn.Linear
    deconv1: eqx.nn.ConvTranspose2d
    deconv2: eqx.nn.ConvTranspose2d
    deconv3: eqx.nn.ConvTranspose2d

    def __init__(self, latent_dim: int, key: PRNGKeyArray):
        keys = jax.random.split(key, 4)
        self.fc = eqx.nn.Linear(latent_dim, 64 * 7 * 7, key=keys[0])
        self.deconv1 = eqx.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            key=keys[1],
        )
        self.deconv2 = eqx.nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding=1,
            key=keys[2],
        )
        self.deconv3 = eqx.nn.ConvTranspose2d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            key=keys[3],
        )

    def __call__(self, z: Float[Array, " lat"]) -> Float[Array, " 1 28 28"]:
        x = jax.nn.relu(self.fc(z))
        x = einops.rearrange(x, "(c h w) -> c h w", c=64, h=7, w=7)
        x = jax.nn.relu(self.deconv1(x))
        x = jax.nn.relu(self.deconv2(x))
        x = jax.nn.sigmoid(self.deconv3(x))
        return x


class VAE(eqx.Module):
    """
    Full VAE combining encoder and decoder networks.
    Uses reparameterization trick to allow backpropagation through sampling.
    """

    encoder: Encoder
    decoder: Decoder
    latent_dim: int

    def __init__(self, latent_dim: int, key: PRNGKeyArray):
        keys = jax.random.split(key, 2)
        self.encoder = Encoder(latent_dim, keys[0])
        self.decoder = Decoder(latent_dim, keys[1])
        self.latent_dim = latent_dim

    def reparameterize(
        self,
        key: PRNGKeyArray,
        mean: Float[Array, " lat"],
        logvar: Float[Array, " lat"],
    ) -> Float[Array, " lat"]:
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mean.shape)
        return mean + eps * std

    def __call__(
        self, key: PRNGKeyArray, x: Float[Array, "1 28 28"]
    ) -> Tuple[
        Float[Array, "1 28 28"], Float[Array, " lat"], Float[Array, " lat"]
    ]:
        mean, logvar = self.encoder(x)
        z = self.reparameterize(key, mean, logvar)
        return self.decoder(z), mean, logvar

    def generate_samples(
        self, key: PRNGKeyArray, num_samples: int = 25
    ) -> Float[Array, "n 1 28 28"]:
        """Generate new images by sampling from the latent space"""
        z = jax.random.normal(key, (num_samples, self.latent_dim))
        return jax.vmap(self.decoder)(z)

    def save(self, filename: str):
        """Save model hyperparameters and weights to a file"""
        with open(filename, "wb") as f:
            # Save hyperparameters as JSON
            hyperparam_str = json.dumps({
                'latent_dim': int(self.latent_dim),
            })
            f.write((hyperparam_str + "\n").encode())
            # Save weights
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(cls, filename: str) -> "VAE":
        """Load model hyperparameters and weights from a file"""
        with open(filename, "rb") as f:
            # Load hyperparameters
            hyperparams = json.loads(f.readline().decode())
            # Create skeleton model with same architecture
            model = cls(
                latent_dim=hyperparams["latent_dim"],
                key=jax.random.PRNGKey(0)  # Key doesn't matter for loading
            )
            # Load weights into skeleton
            return eqx.tree_deserialise_leaves(f, model)

def compute_loss(
    model: VAE, key: PRNGKeyArray, x: Float[Array, "1 28 28"]
) -> Float[Array, ""]:
    """
    VAE loss function = reconstruction loss + KL divergence
    - Reconstruction measures how well we can rebuild the input
    - KL divergence pushes the latent distribution toward standard normal
    """
    x = x.astype(jnp.float32)
    recon_x, mean, logvar = model(key, x)

    # Reconstruction loss (binary cross entropy)
    recon_loss = -jnp.mean(
        x * jnp.log(recon_x + 1e-8) + (1 - x) * jnp.log(1 - recon_x + 1e-8)
    )

    # KL divergence between q(z|x) and p(z) = N(0,1)
    kl_loss = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))

    # Balance reconstruction vs KL terms
    beta = 0.1  # Weight for KL term
    return recon_loss + beta * kl_loss


def vis_samples(
    samples: Float[Array, "n 1 28 28"], columns: int = 2
) -> mp.plot:
    samples = einops.rearrange(samples, "n 1 h w -> n h w")
    num_samples = len(samples)
    rows = (num_samples + columns - 1) // columns  # Ceiling division
    plots = []
    for i in range(rows):
        row = []
        for j in range(columns):
            idx = i * columns + j
            if idx < num_samples:
                img = jnp.clip(samples[idx], 0, 1)
                img_plot = mp.border(mp.image(img))
                row.append(img_plot)
        if row:
            plots.append(mp.hstack(*row))
    return mp.vstack(*plots)
