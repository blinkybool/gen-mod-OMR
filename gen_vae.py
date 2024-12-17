"""
Generate samples from a trained VAE model.
"""


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from vae import VAE, vis_samples


def interpolate(
    model: VAE,
    z1: Float[Array, " lat"],
    z2: Float[Array, " lat"],
    num_steps: int = 10,
) -> Float[Array, "n 1 28 28"]:
    """Interpolate between two points in latent space"""
    alphas = jnp.linspace(0, 1, num_steps)
    z_interp = jnp.stack([(1 - alpha) * z1 + alpha * z2 for alpha in alphas])
    return jax.vmap(model.decoder)(z_interp)


def main(
    model_path: str = "vae_final.eqx",
    num_samples: int = 9,
    seed: int = 42,
    mode: str = "random",  # "random" or "interpolate"
):
    key = jax.random.key(seed)
    key, model_key = jax.random.split(key)

    # Load model
    print(f"Loading model from {model_path}...")
    model = VAE.load(model_path)
    latent_dim = model.latent_dim  # Get latent_dim from the loaded model

    if mode == "random":
        # Generate random samples
        key, sample_key = jax.random.split(key)
        samples = jax.vmap(model.decoder)(
            jax.random.normal(sample_key, (num_samples, latent_dim))
        )
        plot = vis_samples(samples, columns=3)
        print("\nRandom samples:")
        print(plot)

    elif mode == "interpolate":
        # Generate interpolation between two random points
        key, z1_key, z2_key = jax.random.split(key, 3)
        z1 = jax.random.normal(z1_key, (latent_dim,))
        z2 = jax.random.normal(z2_key, (latent_dim,))
        samples = interpolate(model, z1, z2)
        plot = vis_samples(samples, columns=3)  # Show all in one row
        print("\nInterpolation:")
        print(plot)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
