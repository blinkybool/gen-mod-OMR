from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import einops
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro
from sklearn.metrics import silhouette_score

from vae import VAE


@dataclass
class LatentDimAnalysis:
    """Analyze models across different latent dimensions"""
    models_dir: Path
    """Directory containing the trained models"""

    dataset: Literal['train', 'test'] = 'test'
    """Which dataset to analyze"""

    save_path: Path | None = None
    """Where to save the plot"""

def compute_reconstruction_loss(model: VAE, data: jnp.ndarray) -> float:
    """Compute average reconstruction loss for a batch of images"""
    # Add channel dimension
    data = einops.rearrange(data, "n h w -> n 1 h w")

    # Generate reconstructions
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, len(data))
    recons, _, _ = jax.vmap(model)(keys, data)

    # Compute MSE
    mse = jnp.mean((data - recons)**2)
    return float(mse)

def compute_kl_divergence(model: VAE, data: jnp.ndarray) -> float:
    """Compute average KL divergence between encoder distribution and prior"""
    data = einops.rearrange(data, "n h w -> n 1 h w")
    means, logvars = jax.vmap(model.encoder)(data)
    # KL divergence between N(μ,σ) and N(0,1)
    kl = -0.5 * jnp.mean(1 + logvars - means**2 - jnp.exp(logvars))
    return float(kl)

def compute_clustering_metrics(model: VAE, data: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Compute silhouette score of digit clusters in latent space"""
    # Get latent representations
    data = einops.rearrange(data, "n h w -> n 1 h w")
    means, _ = jax.vmap(model.encoder)(data)

    # Convert to numpy for sklearn
    means = np.array(means)
    labels = np.array(labels)

    # Compute silhouette score (higher = better separated clusters)
    return float(silhouette_score(means, labels))

def main(args: LatentDimAnalysis):
    # Load dataset
    with np.load("mnist.npz") as data:
        x_data = data[f"x_{args.dataset}"].astype("float32") / 255.0
        y_data = data[f"y_{args.dataset}"]

    # Collect results for each latent dimension
    results = []
    for model_path in sorted(args.models_dir.glob("vae_latent_dim_*.eqx")):
        # Extract latent dim from filename
        latent_dim = int(model_path.stem.split("_")[-1])
        print(f"Processing model with latent_dim={latent_dim}")

        # Load model and compute metrics
        model = VAE.load(str(model_path))
        metrics = {
            "latent_dim": latent_dim,
            "recon_loss": compute_reconstruction_loss(model, x_data),
            "kl_divergence": compute_kl_divergence(model, x_data),
            "silhouette_score": compute_clustering_metrics(model, x_data, y_data)
        }
        results.append(metrics)

    # Sort by latent dimension
    results.sort(key=lambda x: x["latent_dim"])

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot reconstruction loss
    ax = axes[0]
    ax.plot([r["latent_dim"] for r in results],
            [r["recon_loss"] for r in results], 'o-')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Loss')
    ax.grid(True)

    # Plot KL divergence
    ax = axes[1]
    ax.plot([r["latent_dim"] for r in results],
            [r["kl_divergence"] for r in results], 'o-')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence to Prior')
    ax.grid(True)

    # Plot silhouette score
    ax = axes[2]
    ax.plot([r["latent_dim"] for r in results],
            [r["silhouette_score"] for r in results], 'o-')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Cluster Separation Quality')
    ax.grid(True)

    plt.suptitle(f'VAE Latent Space Analysis ({args.dataset} set)')
    plt.tight_layout()

    if args.save_path:
        plt.savefig(args.save_path)
    else:
        plt.show()
    plt.close()

    # Print numerical results
    print("\nNumerical Results:")
    print("Latent Dim | Recon Loss | KL Divergence | Silhouette")
    print("-" * 55)
    for r in results:
        print(f"{r['latent_dim']:10d} | {r['recon_loss']:.6f} | {r['kl_divergence']:.6f} | {r['silhouette_score']:.6f}")

if __name__ == "__main__":
    args = tyro.cli(LatentDimAnalysis)
    main(args)
