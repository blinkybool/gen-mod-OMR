"""
Generate samples from a trained VAE model.
"""


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import einops

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
    mode: str = "random",  # "random", "interpolate", or "reconstruct"
    output_path: Path | None = None,
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

    elif mode == "reconstruct":
        # Load and reconstruct MNIST test images
        with np.load("mnist.npz") as data:
            x_test = jnp.array(data["x_test"].astype("float32") / 255.0)
        x_test = x_test.reshape(-1, 1, 28, 28)

        # Take first num_samples test images
        test_images = x_test[:num_samples]

        # Generate reconstructions
        key, recon_key = jax.random.split(key)
        recons, _, _ = jax.vmap(lambda x: model(recon_key, x))(test_images)

        # Stack originals and reconstructions
        samples = jnp.concatenate([test_images, recons])
        plot = vis_samples(samples, columns=num_samples)
        print("\nReconstructions (top: original, bottom: reconstructed):")
        print(plot)

        if output_path is not None:
            n=num_samples
            # Create a figure with 2 rows of subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2*n, 4))
            fig.suptitle('VAE Reconstruction Results', fontsize=14, y=0.95)

            # Split original and reconstructed images
            originals = samples[:num_samples].squeeze(axis=1)
            reconstructions = samples[num_samples:].squeeze(axis=1)

            # Create image grids
            orig_grid = einops.rearrange(originals, 'n h w -> h (n w)', n=n)
            recon_grid = einops.rearrange(reconstructions, 'n h w -> h (n w)', n=n)

            # Plot with titles and proper spacing
            ax1.imshow(orig_grid, cmap='gray')
            ax1.set_title('Original Images', pad=10, fontsize=12)
            ax1.axis('off')

            ax2.imshow(recon_grid, cmap='gray')
            ax2.set_title('Reconstructed Images', pad=10, fontsize=12)
            ax2.axis('off')

            # Adjust spacing between subplots
            plt.tight_layout()

            # Save with high DPI for better quality
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=150)
            plt.close()

if __name__ == "__main__":
    import tyro

    tyro.cli(main)
