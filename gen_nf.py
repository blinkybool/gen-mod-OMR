import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import einops

from normalizing_flow import NormalizingFlow
from vae import vis_samples  # Reuse your visualization function

def main(
    model_path: str = "nf_final.eqx",
    num_samples: int = 9,
    seed: int = 42,
    mode: str = "random",  # "random" or "reconstruct"
    output_path: Path | None = None,
):
    key = jax.random.key(seed)

    print(f"Loading model from {model_path}...")
    model = NormalizingFlow.load(model_path)

    if mode == "random":
        samples = model.sample(key, shape=(num_samples, 1, 28, 28))
        plot = vis_samples(samples, columns=3)
        print("\nRandom samples:")
        print(plot)

    elif mode == "reconstruct":
        with np.load("mnist.npz") as data:
            x_test = jnp.array(data["x_test"].astype("float32") / 255.0)
        x_test = x_test.reshape(-1, 1, 28, 28)

        test_images = x_test[:num_samples]

        # For flows, reconstruction means forward then inverse
        z, _ = model.forward(test_images)
        recons = model.inverse(z)

        samples = jnp.concatenate([test_images, recons])
        plot = vis_samples(samples, columns=num_samples)
        print("\nReconstructions (top: original, bottom: reconstructed):")
        print(plot)

        if output_path is not None:
            n = num_samples
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2*n, 4))
            fig.suptitle('NF Reconstruction Results', fontsize=14, y=0.95)

            originals = samples[:num_samples].squeeze(axis=1)
            reconstructions = samples[num_samples:].squeeze(axis=1)

            orig_grid = einops.rearrange(originals, 'n h w -> h (n w)', n=n)
            recon_grid = einops.rearrange(reconstructions, 'n h w -> h (n w)', n=n)

            ax1.imshow(orig_grid, cmap='gray')
            ax1.set_title('Original Images', pad=10, fontsize=12)
            ax1.axis('off')

            ax2.imshow(recon_grid, cmap='gray')
            ax2.set_title('Reconstructed Images', pad=10, fontsize=12)
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=150)
            plt.close()

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
