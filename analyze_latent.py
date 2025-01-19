import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import einops
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tyro
from jaxtyping import Array, Float, Int
from sklearn.decomposition import PCA

from vae import VAE


@dataclass
class LatentAnalysis:
    """Analyze the latent space of a trained VAE."""
    model_path: Path
    """Path to the trained VAE model."""

    mode: Literal['distance_matrix', 'reconstruction', 'interpolation', 'pca']
    """Analysis mode to run."""

    dataset: Literal['train', 'test'] = 'test'
    """Whether to analyze training or test set"""

    num_samples: int | None = None
    """Number of samples to use"""

    digits: list[int] | None = None
    """Which digits to show in visualization (None means show all)"""

    num_steps: int = 8
    """Number of interpolation steps"""

    num_pairs: int = 5
    """Number of different pairs to interpolate between"""

    save_path: Path | None = None
    """Optional path to save visualizations."""

def encode_dataset(model: VAE, data: Float[Array, "n h w"]) -> tuple[Float[Array, "n lat"], Float[Array, "n lat"]]:
    """Encode dataset to get latent representations"""
    batched_data = einops.rearrange(data, "n h w -> n 1 h w")
    means_logvars = jax.vmap(model.encoder)(batched_data)
    return means_logvars

def compute_distance_matrix(latent_means: Float[Array, "n lat"], labels: Int[Array, "n"]) -> Float[Array, "n_digits n_digits"]:
    """Compute average distance between digit clusters in latent space."""
    n_digits = 10
    distance_matrix = jnp.zeros((n_digits, n_digits))

    # Compute cluster centers
    centers = []
    for digit in range(n_digits):
        mask = labels == digit
        center = latent_means[mask].mean(axis=0)
        centers.append(center)
    centers = jnp.stack(centers)

    # Compute pairwise distances
    for i in range(n_digits):
        for j in range(n_digits):
            distance_matrix = distance_matrix.at[i, j].set(
                jnp.linalg.norm(centers[i] - centers[j])
            )

    return distance_matrix

def plot_heatmap(matrix: Float[Array, "n_digits n_digits"], title: str, save_path: Path | None = None):
    """Plot a heatmap with digit pairs."""
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)])
    ax.invert_yaxis()
    plt.title(title)
    plt.xlabel('Digit')
    plt.ylabel('Digit')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def analyze_reconstruction_quality(model: VAE, data: Float[Array, "n h w"], labels: Int[Array, "n"]) -> list[float]:
    """Analyze reconstruction quality per digit."""
    n_digits = 10
    recon_errors = []

    for digit in range(n_digits):
        mask = labels == digit
        digit_data = data[mask]

        # Add batch dimension and compute reconstructions with vmap
        batched_data = einops.rearrange(digit_data, "n h w -> n 1 h w")
        keys = jax.random.split(jax.random.PRNGKey(0), len(digit_data))
        recons, _, _ = jax.vmap(model)(keys, batched_data)

        # Compute MSE for this digit
        mse = jnp.mean((digit_data - einops.rearrange(recons, "n 1 h w -> n h w"))**2)
        recon_errors.append(float(mse))

    return recon_errors

def plot_pca_views(latent_dim: int, latent_means: Float[Array, "n lat"], labels: Int[Array, "n"],
                  digits_to_show: list[int] | None = None) -> plt.Figure:
    """Create PCA visualization of first two components."""
    # Filter data if specific digits requested
    if digits_to_show is not None:
        mask = sum(labels == d for d in digits_to_show).astype(bool)
        latent_means = latent_means[mask]
        labels = labels[mask]

    # Fit PCA
    pca = PCA()
    pca_result = pca.fit_transform(np.array(latent_means))

    # Create figure
    fig = plt.figure(figsize=(8, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Plot each digit
    unique_digits = np.unique(labels)
    for digit in unique_digits:
        mask = labels == digit
        plt.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            c=[colors[digit]],
            label=f'Digit {digit}',
            alpha=0.6
        )

    plt.title(f'PCA View of VAE Latent Space for MNIST dataset\nLatent dim={latent_dim}, n={len(labels)}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()

    return fig

def interpolate_digits(
    model: VAE,
    data: Float[Array, "n h w"],
    labels: Int[Array, "n"],
    digit1: int,
    digit2: int,
    num_steps: int = 10,
    num_pairs: int = 3,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
) -> plt.Figure:
    """Visualize interpolations between pairs of digits in latent space."""
    # Get random samples of each digit
    mask1 = labels == digit1
    mask2 = labels == digit2
    idx1 = jax.random.choice(key, jnp.where(mask1)[0], shape=(num_pairs,))
    idx2 = jax.random.choice(key, jnp.where(mask2)[0], shape=(num_pairs,))

    # Set up the plot
    fig, axes = plt.subplots(num_pairs, num_steps + 2,
                            figsize=((num_steps + 2) * 1.5, num_pairs * 1.5))

    for pair in range(num_pairs):
        # Get start and end images
        start_img = data[idx1[pair]]
        end_img = data[idx2[pair]]

        # Encode images
        start_mean, _ = model.encoder(einops.rearrange(start_img, "h w -> 1 h w"))
        end_mean, _ = model.encoder(einops.rearrange(end_img, "h w -> 1 h w"))

        # Generate interpolation steps
        alphas = jnp.linspace(0, 1, num_steps)
        interpolated_z = jnp.array([
            (1 - alpha) * start_mean + alpha * end_mean
            for alpha in alphas
        ])

        # Decode interpolated points
        interpolated_x = jax.vmap(model.decoder)(interpolated_z)

        # Plot original start image
        axes[pair, 0].imshow(start_img, cmap='gray')
        axes[pair, 0].axis('off')
        if pair == 0:
            axes[pair, 0].set_title(f'{digit1}')

        # Plot interpolation steps
        for step in range(num_steps):
            axes[pair, step + 1].imshow(
                einops.rearrange(interpolated_x[step], "1 h w -> h w"),
                cmap='gray'
            )
            axes[pair, step + 1].axis('off')

        # Plot original end image
        axes[pair, -1].imshow(end_img, cmap='gray')
        axes[pair, -1].axis('off')
        if pair == 0:
            axes[pair, -1].set_title(f'{digit2}')

    plt.suptitle(f'Interpolation between digits {digit1} and {digit2}')
    plt.tight_layout()
    return fig

def get_balanced_subset(x: Float[Array, "n h w"], y: Int[Array, "n"], samples_per_digit: int,
                       key: Array = jax.random.PRNGKey(0)) -> tuple[Float[Array, "m h w"], Int[Array, "m"]]:
    """Get a balanced subset of samples for each digit using JAX."""
    key, subkey = jax.random.split(key)
    indices = []

    # Get samples for each digit
    for digit in range(10):
        digit_indices = jnp.where(y == digit)[0]
        digit_key, key = jax.random.split(key)
        selected = jax.random.choice(digit_key, digit_indices, shape=(samples_per_digit,), replace=False)
        indices.append(selected)

    # Concatenate and shuffle
    indices = jnp.concatenate(indices)
    indices = jax.random.permutation(subkey, indices)

    return x[indices], y[indices]

def analyze_latent(args: LatentAnalysis) -> None:
    # Load model and data
    model = VAE.load(str(args.model_path))
    print(f"\nAnalyzing VAE with latent dimension: {model.latent_dim}")

    with np.load("mnist.npz") as data:
        x_data = data[f"x_{args.dataset}"].astype("float32") / 255.0
        y_data = data[f"y_{args.dataset}"]

    # Get balanced subset if num_samples specified
    if args.num_samples is not None:
        samples_per_digit = args.num_samples // 10
        x_data, y_data = get_balanced_subset(x_data, y_data, samples_per_digit)
        print(f"Using {len(y_data)} samples ({samples_per_digit} per digit)")

    # Get latent representations
    latent_vectors = encode_dataset(model, x_data)
    latent_means = latent_vectors[0]  # First element is mean

    if args.mode == "distance_matrix":
        distance_matrix = compute_distance_matrix(latent_means, y_data)
        plot_heatmap(distance_matrix,
                    f"VAE Latent Space Cluster Center Distances (dim={model.latent_dim})",
                    args.save_path)

        # Additional analysis of distances
        print("\nLatent Space Distance Analysis:")
        print("- Most similar digit pairs:")
        flat_distances = []
        for i in range(10):
            for j in range(i+1, 10):
                flat_distances.append((i, j, distance_matrix[i, j]))
        for i, j, dist in sorted(flat_distances, key=lambda x: x[2])[:5]:
            print(f"  {i}-{j}: {dist:.3f}")

        print("\n- Most dissimilar digit pairs:")
        for i, j, dist in sorted(flat_distances, key=lambda x: x[2], reverse=True)[:5]:
            print(f"  {i}-{j}: {dist:.3f}")

    elif args.mode == "reconstruction":
        recon_errors = analyze_reconstruction_quality(model, x_data, y_data)
        print("\nReconstruction Quality Analysis:")
        print("Per-digit MSE:")
        for digit, error in enumerate(recon_errors):
            print(f"Digit {digit}: {error:.4f}")

        # Plot reconstruction errors
        plt.figure(figsize=(10, 6))
        plt.bar(range(10), recon_errors)
        plt.title(f"Reconstruction MSE by Digit (dim={model.latent_dim})")
        plt.xlabel("Digit")
        plt.ylabel("MSE")
        if args.save_path:
            plt.savefig(args.save_path)
        else:
            plt.show()
        plt.close()

    elif args.mode == "interpolation":
        if args.digits is None or len(args.digits) != 2:
            raise ValueError("Must specify exactly two digits for interpolation mode")

        # Interpolate between consecutive pairs
        digit1, digit2 = args.digits[0], args.digits[1]
        print(f"\nInterpolating {digit1} → {digit2}")
        fig = interpolate_digits(
            model, x_data, y_data,
            digit1, digit2,
            num_steps=args.num_steps,
            num_pairs=args.num_pairs,
            key=jax.random.PRNGKey(int(time.time() * 1000))
        )

        if args.save_path:
            # Modify save path to include digit pair
            stem = args.save_path.stem
            suffix = args.save_path.suffix
            pair_path = args.save_path.with_name(f"{stem}_{digit1}to{digit2}{suffix}")
            fig.savefig(pair_path)
        else:
            plt.show()
        plt.close(fig)

    elif args.mode == "pca":
        fig = plot_pca_views(model.latent_dim, latent_means, y_data, args.digits)
        if args.save_path:
            fig.savefig(args.save_path)
        else:
            plt.show()
        plt.close(fig)

if __name__ == "__main__":
    args = tyro.cli(LatentAnalysis)
    analyze_latent(args)
