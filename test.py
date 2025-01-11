from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from normalizing_flow import NormalizingFlow


def load_single_mnist_image() -> jnp.ndarray:
    """Load single MNIST image with shape (1,28,28)"""
    with np.load("mnist.npz") as data:
        x = jnp.array(data["x_train"][0].astype("float32") / 255.0)
    x = x.reshape(1, 28, 28)  # Single channel image
    x = 2 * x - 1  # Scale to [-1, 1]
    return x

def save_images(imgs: jnp.ndarray, path: str, title: str = None):
    """
    Save multiple images in a row
    Args:
        imgs: Array of shape (N,1,28,28) in range [-1,1]
    """
    # Scale from [-1,1] to [0,1] for display
    imgs = (imgs + 1) / 2
    imgs = np.clip(imgs, 0, 1)

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    if n == 1:
        axes = [axes]
    if title:
        fig.suptitle(title)

    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].squeeze(), cmap='gray')
        ax.axis('off')

    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def test_single_transform():
    """Test single image transformation"""
    print("\nTesting single image transform...")

    # Initialize model
    key = jax.random.key(42)
    model = NormalizingFlow(n_layers=8, channels=1, key=key)

    # Load single image
    x = load_single_mnist_image()
    print("Input shape:", x.shape)

    # Forward pass
    z, log_det = model.forward(x)
    print("Forward shape:", z.shape)
    print("Log det:", log_det)

    # Inverse pass
    x_recon = model.inverse(z)
    print("Reconstruction shape:", x_recon.shape)
    print("Reconstruction error:", jnp.abs(x - x_recon).mean())

    # Save results
    save_images(x[None], "test_outputs/original.png", "Original")
    save_images(x_recon[None], "test_outputs/reconstruction.png", "Reconstruction")

def test_batch_transform():
    """Test batched transformation using vmap"""
    print("\nTesting batched transform...")

    # Initialize model
    key = jax.random.key(42)
    model = NormalizingFlow(n_layers=8, channels=1, key=key)

    # Load batch of images
    with np.load("mnist.npz") as data:
        x_batch = jnp.array(data["x_train"][:5].astype("float32") / 255.0)
    x_batch = x_batch.reshape(-1, 1, 28, 28)
    x_batch = 2 * x_batch - 1
    print("Batch input shape:", x_batch.shape)

    # Create batched versions of forward and inverse
    batch_forward = jax.vmap(model.forward)
    batch_inverse = jax.vmap(model.inverse)

    # Forward pass
    z_batch, log_dets = batch_forward(x_batch)
    print("Batch forward shape:", z_batch.shape)
    print("Batch log dets shape:", log_dets.shape)

    # Inverse pass
    x_recon_batch = batch_inverse(z_batch)
    print("Batch reconstruction shape:", x_recon_batch.shape)
    print("Batch reconstruction error:", jnp.abs(x_batch - x_recon_batch).mean())

    # Save results
    save_images(x_batch, "test_outputs/batch_original.png", "Batch Original")
    save_images(x_recon_batch, "test_outputs/batch_reconstruction.png", "Batch Reconstruction")

def test_sampling():
    """Test random sampling"""
    print("\nTesting sampling...")

    # Initialize model
    key = jax.random.key(42)
    model = NormalizingFlow(n_layers=8, channels=1, key=key)

    # Single sample
    sample_key = jax.random.key(0)
    sample = model.sample(sample_key, shape=(1, 28, 28))
    print("Single sample shape:", sample.shape)

    # Batch of samples
    batch_sample = jax.vmap(lambda k: model.sample(k, (1, 28, 28)))
    keys = jax.random.split(sample_key, 5)
    samples = batch_sample(keys)
    print("Batch samples shape:", samples.shape)

    # Save results
    save_images(samples, "test_outputs/samples.png", "Random Samples")

def main():
    # Create output directory
    Path("test_outputs").mkdir(exist_ok=True)

    # Run tests
    test_single_transform()
    test_batch_transform()
    test_sampling()

if __name__ == "__main__":
    main()
