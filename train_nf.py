from pathlib import Path
from typing import Optional, Tuple

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm
from jaxtyping import Array, Float, PRNGKeyArray

import wandb
from normalizing_flow import NormalizingFlow


def save_image_grid(images: Float[Array, "batch 1 28 28"], path: Path, title: Optional[str] = None):
    """Save a grid of images
    Args:
        images: Array of shape (N,1,28,28) in range [-1,1]
    """
    # Scale from [-1,1] to [0,1] for display
    display_images = np.array(images)
    display_images = (display_images + 1) / 2
    display_images = np.clip(display_images, 0, 1)

    n = len(display_images)
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    if n == 1:
        axes = [axes]
    if title:
        fig.suptitle(title)

    for i, ax in enumerate(axes):
        ax.imshow(display_images[i].squeeze(), cmap='gray')
        ax.axis('off')

    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

@jax.jit
def compute_batch_loss(model: NormalizingFlow, batch: Float[Array, "batch 1 28 28"]) -> Float[Array, ""]:
    """Compute average negative log likelihood for a batch"""
    return -jnp.mean(jax.vmap(model.log_prob)(batch))

@jax.jit
def train_step(
    model: NormalizingFlow,
    opt_state: optax.OptState,
    batch: Float[Array, "batch 1 28 28"],
    optimizer: optax.GradientTransformation = None,  # Will be filled in with partial
) -> Tuple[NormalizingFlow, optax.OptState, Float[Array, ""]]:
    """Single training step"""
    def loss_fn(m):
        return compute_batch_loss(m, batch)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

@jax.jit
def generate_reconstructions(
    model: NormalizingFlow,
    batch: Float[Array, "batch 1 28 28"]
) -> Float[Array, "batch 1 28 28"]:
    """Generate reconstructions for a batch of images"""
    forward_fn = jax.vmap(model.forward)
    inverse_fn = jax.vmap(model.inverse)
    z, _ = forward_fn(batch)
    return inverse_fn(z)

def save_training_images(
    model: NormalizingFlow,
    batch: Float[Array, "batch 1 28 28"],
    key: PRNGKeyArray,
    output_dir: Path,
    epoch: int,
    use_wandb: bool = False
):
    """Save/log training images"""
    # Generate reconstructions
    recons = generate_reconstructions(model, batch)

    # Generate samples
    samples = jax.vmap(lambda k: model.sample(k, (1,28,28)))(
        jax.random.split(key, batch.shape[0])
    )

    # Save to files
    save_image_grid(batch, output_dir / f"epoch_{epoch}_originals.png", "Originals")
    save_image_grid(recons, output_dir / f"epoch_{epoch}_reconstructions.png", "Reconstructions")
    save_image_grid(samples, output_dir / f"epoch_{epoch}_samples.png", "Samples")

    if use_wandb:
        wandb.log({
            "reconstructions": [
                wandb.Image(np.hstack([
                    np.array(orig).squeeze(),  # Explicit copy
                    np.ones((28, 4)),
                    np.array(recon).squeeze()  # Explicit copy
                ]))
                for orig, recon in zip(batch, recons)
            ],
            "samples": [wandb.Image(np.array(img).squeeze()) for img in samples],
        })

def main(
    n_layers: int = 8,
    learning_rate: float = 1e-4,
    batch_size: int = 256,
    num_epochs: int = 50,
    num_vis_samples: int = 10,
    seed: int = 42,
    output_dir: Path = Path("runs/nf"),
    use_wandb: bool = False,
    wandb_project: Optional[str] = "nf-mnist",
    wandb_entity: Optional[str] = None,
):
    """Train normalizing flow model"""
    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(seed)

    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "n_layers": n_layers,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "seed": seed,
            }
        )
        run_dir = output_dir / run.id
        run_dir.mkdir(exist_ok=True)
    else:
        run_dir = output_dir / "test_run"
        run_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading mnist.npz...")
    with np.load("mnist.npz") as data:
        x_train = jnp.array(data["x_train"].astype("float32") / 255.0)
        x_test = jnp.array(data["x_test"].astype("float32") / 255.0)

    # Preprocess
    x_train = 2 * x_train - 1  # Scale to [-1, 1]
    x_test = 2 * x_test - 1
    x_train = einops.rearrange(x_train, "b h w -> b 1 h w")
    x_test = einops.rearrange(x_test, "b h w -> b 1 h w")

    num_batches = len(x_train) // batch_size
    print(f"Training on {len(x_train)} images with {num_batches} batches per epoch")

    # Initialize model and optimizer
    print("Initializing model and optimizer...")
    key, model_key = jax.random.split(key)
    model = NormalizingFlow(n_layers=n_layers, channels=1, key=model_key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @jax.jit
    def train_step(model, opt_state, batch):
        def loss_fn(m):
            return compute_batch_loss(m, batch)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Training loop
    print("Beginning training...")
    for epoch in range(num_epochs):
        key, shuffle_key = jax.random.split(key)
        x_train_shuffled = x_train[jax.random.permutation(shuffle_key, len(x_train))]

        # Train for one epoch
        epoch_losses = []
        pbar = tqdm.tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = start + batch_size
            batch = x_train_shuffled[start:end]

            model, opt_state, loss = train_step(model, opt_state, batch)
            epoch_losses.append(float(loss))
            pbar.set_postfix({"loss": f"{float(loss):.4f}"})

            if use_wandb:
                wandb.log({
                    "training/batch_loss": float(loss),
                    "training/epoch": epoch,
                    "training/batch": batch_idx,
                    "training/progress": (epoch * num_batches + batch_idx) / (num_epochs * num_batches)
                })

        # Evaluation and logging
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        val_loss = compute_batch_loss(model, x_test)
        print(f"Epoch {epoch+1}: train_loss={avg_epoch_loss:.4f}, val_loss={float(val_loss):.4f}")

        if use_wandb:
            wandb.log({
                "epoch/train_loss": avg_epoch_loss,
                "epoch/val_loss": float(val_loss),
                "epoch": epoch,
            })

        # Save images
        key, vis_key = jax.random.split(key)
        vis_batch = x_train_shuffled[:num_vis_samples]
        save_training_images(model, vis_batch, vis_key, run_dir, epoch, use_wandb)

        # Save model checkpoint
        if use_wandb:
            checkpoint_path = run_dir / f"nf_checkpoint_epoch_{epoch+1}.eqx"
            model.save(checkpoint_path)
            artifact = wandb.Artifact(
                name=f"model-epoch-{epoch+1}",
                type="model",
                description=f"Model checkpoint at epoch {epoch+1}"
            )
            artifact.add_file(str(checkpoint_path))
            run.log_artifact(artifact)

    # Save final model
    final_path = run_dir / "nf_final.eqx"
    model.save(final_path)
    if use_wandb:
        artifact = wandb.Artifact(
            name="model-final",
            type="model",
            description="Final trained model"
        )
        artifact.add_file(str(final_path))
        run.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
