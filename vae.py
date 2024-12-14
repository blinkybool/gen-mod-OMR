"""
Variational Autoencoder (VAE) for MNIST digit generation, implemented with JAX.
"""
from typing import Tuple
from jaxtyping import Array, Float, PRNGKeyArray

import jax
import jax.numpy as jnp
import einops
import optax
from strux import struct

import tqdm
import mattplotlib as mp
import numpy as np
import equinox as eqx

class Encoder(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc_mu: eqx.nn.Linear
    fc_logvar: eqx.nn.Linear

    def __init__(self, latent_dim: int, key: PRNGKeyArray):
        keys = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, key=keys[1])
        self.fc_mu = eqx.nn.Linear(64 * 7 * 7, latent_dim, key=keys[2])
        self.fc_logvar = eqx.nn.Linear(64 * 7 * 7, latent_dim, key=keys[3])

    def __call__(self, x: Float[Array, "1 28 28"]) -> Tuple[Float[Array, "lat"], Float[Array, "lat"]]:
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = einops.rearrange(x, 'c h w -> (c h w)')
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(eqx.Module):
    fc: eqx.nn.Linear
    deconv1: eqx.nn.ConvTranspose2d
    deconv2: eqx.nn.ConvTranspose2d
    deconv3: eqx.nn.ConvTranspose2d

    def __init__(self, latent_dim: int, key: PRNGKeyArray):
        keys = jax.random.split(key, 4)
        self.fc = eqx.nn.Linear(latent_dim, 64 * 7 * 7, key=keys[0])
        self.deconv1 = eqx.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, key=keys[1])
        self.deconv2 = eqx.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, key=keys[2])
        self.deconv3 = eqx.nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, key=keys[3])

    def __call__(self, z: Float[Array, "lat"]) -> Float[Array, "1 28 28"]:
        x = jax.nn.relu(self.fc(z))
        x = einops.rearrange(x, '(c h w) -> c h w', c=64, h=7, w=7)
        x = jax.nn.relu(self.deconv1(x))
        x = jax.nn.relu(self.deconv2(x))
        x = jax.nn.sigmoid(self.deconv3(x))
        return x

class VAE(eqx.Module):
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
        mean: Float[Array, "lat"],
        logvar: Float[Array, "lat"]
    ) -> Float[Array, "lat"]:
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mean.shape)
        return mean + eps * std

    def __call__(
        self,
        key: PRNGKeyArray,
        x: Float[Array, "1 28 28"]
    ) -> Tuple[Float[Array, "1 28 28"], Float[Array, "lat"], Float[Array, "lat"]]:
        mean, logvar = self.encoder(x)
        z = self.reparameterize(key, mean, logvar)
        return self.decoder(z), mean, logvar

def compute_loss(
    model: VAE,
    key: PRNGKeyArray,
    x: Float[Array, "1 28 28"]
) -> float:
    x = x.astype(jnp.float32)
    recon_x, mean, logvar = model(key, x)

    # Reconstruction loss (binary cross entropy)
    recon_loss = -jnp.sum(
        x * jnp.log(recon_x + 1e-8) + (1 - x) * jnp.log(1 - recon_x + 1e-8)
    )

    # KL divergence
    kl_loss = -0.5 * jnp.sum(1 + logvar - mean**2 - jnp.exp(logvar))

    return recon_loss + kl_loss

def main(
    latent_dim: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    num_epochs: int = 50,
    batches_per_visual: int = 10,
    seed: int = 42,
):
    key = jax.random.key(seed)

    print("initialising model...")
    with np.load('mnist.npz') as data:
        x_train = jnp.array(data['x_train'].astype('float32') / 255.)
    x_train = einops.rearrange(x_train, 'b h w -> b 1 h w')

    print("initialising optimiser...")
    key, model_key = jax.random.split(key)
    model = VAE(latent_dim=latent_dim, key=model_key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Process multiple batches at once
    @jax.jit
    def train_batches(model, opt_state, key, batches):
        keys = jax.random.split(key, len(batches))
        def step(carry, inputs):
            model, opt_state = carry
            batch, key = inputs
            model, opt_state, loss = train_step_batch(model, opt_state, key, batch)
            return (model, opt_state), loss
        (model, opt_state), losses = jax.lax.scan(step, (model, opt_state), (batches, keys))
        return model, opt_state, jnp.mean(losses)

    @jax.jit
    def train_step_batch(model, opt_state, key, batch):
        keys = jax.random.split(key, batch.shape[0])
        loss, grads = jax.vmap(train_step_single, in_axes=(None, 0, 0))(model, batch, keys)
        loss = jnp.mean(loss)
        grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss

    @jax.jit
    def train_step_single(model, x, key):
        loss_fn = lambda m: compute_loss(m, key, x)
        return eqx.filter_value_and_grad(loss_fn)(model)

    print("begin training...")
    num_batches = len(x_train) // batch_size
    samples_plot = None

    for epoch in range(num_epochs):
        # Shuffle once per epoch
        key, shuffle_key = jax.random.split(key)
        x_train_shuffled = x_train[jax.random.permutation(shuffle_key, len(x_train))]

        pbar = tqdm.tqdm(range(0, num_batches, batches_per_visual),
                        desc=f"Epoch {epoch+1}/{num_epochs}")

        for i in pbar:
            # Get multiple batches
            start = i * batch_size
            end = start + batches_per_visual * batch_size
            batches = x_train_shuffled[start:end].reshape(-1, batch_size, 1, 28, 28)

            # Train on all batches
            key, train_key = jax.random.split(key)
            model, opt_state, loss = train_batches(model, opt_state, train_key, batches)
            pbar.set_postfix({'loss': float(loss)})

            # Visualize
            key, sample_key = jax.random.split(key)
            samples = generate_samples(model, sample_key, num_samples=9)
            plot = vis_samples(samples, grid_size=3)
            if samples_plot is None:
                tqdm.tqdm.write(str(plot))
                samples_plot = plot
            else:
                tqdm.tqdm.write(f"\x1b[{samples_plot.height}A{plot}")
                samples_plot = plot

def generate_samples(
    model: VAE,
    key: PRNGKeyArray,
    num_samples: int = 25
) -> Float[Array, "n 1 28 28"]:
    z = jax.random.normal(key, (num_samples, model.latent_dim))
    return jax.vmap(model.decoder)(z)

def vis_samples(samples: Float[Array, "n 1 28 28"], grid_size: int = 3) -> mp.plot:
    samples = einops.rearrange(samples, 'n 1 h w -> n h w')
    plots = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(samples):
                img = jnp.clip(samples[idx], 0, 1)
                img_plot = mp.border(mp.image(img))
                row.append(img_plot)
        if row:
            plots.append(mp.hstack(*row))
    return mp.vstack(*plots)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
