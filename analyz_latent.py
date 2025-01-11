"""
Analyze the latent space of a trained VAE using K-means clustering and PCA visualization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.metrics
import matplotlib.pyplot as plt
import einops
from typing import Tuple, Optional
from jaxtyping import Array, Float, Int

from vae import VAE

def encode_dataset(
    model: VAE,
    data: jnp.ndarray,
    batch_size: int = 128,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Encode the entire dataset into the latent space.
    Returns means and log variances of the latent representations.
    """
    # Ensure data is in correct format (B, 1, H, W)
    if len(data.shape) == 3:
        data = einops.rearrange(data, "b h w -> b 1 h w")

    # Use vmap over the entire dataset
    means, logvars = jax.vmap(model.encoder)(data)
    return means, logvars

def perform_clustering(
    latent_means: jnp.ndarray,
    n_clusters: int = 10,
    random_state: int = 42
) -> Tuple[KMeans, jnp.ndarray]:
    """
    Perform K-means clustering on the latent representations.
    Returns the fitted KMeans model and cluster assignments.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_assignments = kmeans.fit_predict(latent_means)
    return kmeans, cluster_assignments

def visualize_latent_space(
    latent_means: jnp.ndarray,
    cluster_assignments: Int[Array, "b"],
    labels: Optional[Int[Array, "b"]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize the latent space using PCA projection to 2D.
    If labels are provided, create two subplots comparing clustering to true labels.
    """
    # Perform PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_means)

    # Set up the plotting
    if labels is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = None

    # Plot clustering results
    scatter1 = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1],
                          c=cluster_assignments, cmap='tab10', alpha=0.6)
    ax1.set_title('Latent Space - K-means Clustering')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    plt.colorbar(scatter1, ax=ax1, label='Cluster Assignment')

    # If true labels are provided, plot them as well
    if labels is not None and ax2 is not None:
        scatter2 = ax2.scatter(latent_2d[:, 0], latent_2d[:, 1],
                             c=labels, cmap='tab10', alpha=0.6)
        ax2.set_title('Latent Space - True Labels')
        ax2.set_xlabel('First Principal Component')
        ax2.set_ylabel('Second Principal Component')
        plt.colorbar(scatter2, ax=ax2, label='True Label')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_cluster_metrics(
    kmeans: KMeans,
    cluster_assignments: Int[Array, "b"],
    labels: Optional[Int[Array, "b"]] = None
) -> dict:
    """
    Compute various metrics to evaluate the clustering quality.
    """
    # Ensure inertia_ is not None and convert to float
    inertia = kmeans.inertia_
    if inertia is None:
        raise ValueError("KMeans inertia_ is None. Ensure clustering was performed.")

    metrics = {
        'inertia': float(inertia),  # Within-cluster sum of squares
        'cluster_sizes': jnp.bincount(cluster_assignments).tolist(),
    }

    if labels is not None:
        # Add metrics comparing to true labels if available
        metrics.update({
            'adjusted_rand_score': float(sklearn.metrics.adjusted_rand_score(labels, cluster_assignments)),
            'nmi_score': float(sklearn.metrics.normalized_mutual_info_score(labels, cluster_assignments))
        })

    return metrics

def main(
    model_path: str,
    n_clusters: int = 10,
    batch_size: int = 256,
    seed: int = 42,
):
    # Load the model
    print(f"Loading model from {model_path}...")
    model = VAE.load(model_path)

    # Load and preprocess MNIST data
    print("Loading MNIST dataset...")
    with np.load("mnist.npz") as data:
        x_train = jnp.array(data["x_train"].astype("float32") / 255.0)
        y_train = jnp.array(data["y_train"])

    # Encode the dataset
    print("Encoding dataset into latent space...")
    latent_means, latent_logvars = encode_dataset(model, x_train, batch_size)

    # Perform clustering
    print("Performing K-means clustering...")
    kmeans, cluster_assignments = perform_clustering(
        latent_means, n_clusters=n_clusters, random_state=seed
    )

    # Visualize results
    print("Generating visualizations...")
    visualize_latent_space(
        latent_means,
        cluster_assignments,
        labels=y_train,
        save_path="latent_space_visualization.png"
    )

    # Compute and print metrics
    metrics = analyze_cluster_metrics(kmeans, cluster_assignments, y_train)
    print("\nClustering Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
