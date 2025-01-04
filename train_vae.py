import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import shutil
from typing import Literal
from dataclasses import dataclass

from vae import VAE, compute_loss, vis_samples

def train(
    latent_dim: int = 32,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    num_epochs: int = 100,
    batches_per_visual: int = 20,
    checkpoint_every: int = 1,
    seed: int = 42,
    output_folder: str = "models"
):
    """Train a new VAE model"""
    key = jax.random.key(seed)

    print("initialising model...")
    with np.load("mnist.npz") as data:
        x_train = jnp.array(data["x_train"].astype("float32") / 255.0)
    x_train = einops.rearrange(x_train, "b h w -> b 1 h w")

    num_batches = len(x_train) // batch_size

    print("initialising optimiser...")
    key, model_key = jax.random.split(key)
    model = VAE(latent_dim=latent_dim, key=model_key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @jax.jit
    def train_step(model, opt_state, key, batch):
        batch = batch.astype(jnp.float32)

        def batch_loss(model):
            keys = jax.random.split(key, batch.shape[0])
            losses = jax.vmap(compute_loss, in_axes=(None, 0, 0))(model, keys, batch)
            return jnp.mean(losses)

        loss, grads = eqx.filter_value_and_grad(batch_loss)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    print("begin training...")

    for epoch in range(num_epochs):
        key, shuffle_key = jax.random.split(key)
        x_train_shuffled = x_train[jax.random.permutation(shuffle_key, len(x_train))]

        pbar = tqdm.tqdm(
            range(num_batches),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        for i in pbar:
            start = i * batch_size
            end = start + batch_size
            batch = x_train_shuffled[start:end]

            key, train_key = jax.random.split(key)
            model, opt_state, loss = train_step(model, opt_state, train_key, batch)
            pbar.set_postfix({"loss": float(loss)})

            if i % batches_per_visual == 0:
                key, sample_key = jax.random.split(key)
                terminal_width = shutil.get_terminal_size().columns
                max_columns = terminal_width // 30
                num_samples = min(max_columns, 10)
                samples = model.generate_samples(sample_key, num_samples=num_samples)
                plot = vis_samples(samples, columns=num_samples)
                print("\nSamples:")
                print(plot)
                print()

        if (epoch + 1) % checkpoint_every == 0:
            model.save(
                f"{output_folder}/vae_checkpoint_epoch_{epoch+1}_latent{latent_dim}_lr{learning_rate}_batch{batch_size}.eqx"
            )

    model.save(
        f"{output_folder}/vae_final_latent{latent_dim}_lr{learning_rate}_batch{batch_size}.eqx",
    )

def evaluate_loss(
    model_path: str,
    batch_size: int = 128,
    seed: int = 42
):
    """Evaluate the loss of a trained VAE model on the test set"""
    print(f"Loading model from {model_path}...")
    model = VAE.load(model_path)

    print("Loading test data...")
    with np.load("mnist.npz") as data:
        x_test = jnp.array(data["x_test"].astype("float32") / 255.0)
    x_test = einops.rearrange(x_test, "b h w -> b 1 h w")

    key = jax.random.key(seed)
    keys = jax.random.split(key, len(x_test))

    print("Computing loss")
    losses = jax.vmap(compute_loss, in_axes=(None, 0, 0))(model, keys, x_test)
    avg_loss = jnp.mean(losses)

    print(f"\nAverage loss on test set: {avg_loss:.4f}")

if __name__ == "__main__":
    import tyro
    tyro.extras.subcommand_cli_from_dict({
        "train": train,
        "loss": evaluate_loss,
    })
