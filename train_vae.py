import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from pathlib import Path

from vae import VAE, compute_loss

def get_platform():
    """Returns the current hardware platform (cpu, gpu, or tpu)"""
    return jax.default_backend()

def configure_platform():
    """Configure JAX for the current platform"""
    platform = get_platform()
    if platform == "tpu":
        # TPU-specific configuration
        print("Using jax TPU")
        jax.config.update("jax_enable_x64", False)  # TPUs work better with float32
    else:
        # CPU/GPU configuration
        pass  # Add any CPU/GPU specific configs here if needed

def main(
    latent_dim: int = 32,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    num_epochs: int = 50,
    num_vis_samples: int = 10,  # Number of samples/reconstructions to visualize
    seed: int = 42,
    output_folder: Path = Path("runs/vae"),
    wandb_project: str = "vae-mnist",
    wandb_entity: str | None = None,
):
    """Train a new VAE model"""

    configure_platform()
    output_folder.mkdir(parents=True, exist_ok=True)
    Path("wandb").mkdir(exist_ok=True)

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "latent_dim": latent_dim,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "seed": seed,
        }
    )

    try:
        # Create run-specific output directory
        run_dir = output_folder / run.id
        run_dir.mkdir(parents=True, exist_ok=True)

        key = jax.random.key(seed)

        print("Loading mnist.npz...")
        with np.load("mnist.npz") as data:
            x_train = jnp.array(data["x_train"].astype("float32") / 255.0)
            x_test = jnp.array(data["x_test"].astype("float32") / 255.0)
        x_train = einops.rearrange(x_train, "b h w -> b 1 h w")
        x_test = einops.rearrange(x_test, "b h w -> b 1 h w")

        num_batches = len(x_train) // batch_size
        print(f"Training on {len(x_train)} images with {num_batches} batches per epoch")

        print("initialising model/optimiser...")
        key, model_key = jax.random.split(key)
        model = VAE(latent_dim=latent_dim, key=model_key)
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        @jax.jit
        def compute_validation_loss(model, key, batch):
            batch = batch.astype(jnp.float32)
            keys = jax.random.split(key, batch.shape[0])
            losses = jax.vmap(compute_loss, in_axes=(None, 0, 0))(model, keys, batch)
            return jnp.mean(losses)

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

            epoch_losses = []
            pbar = tqdm.tqdm(
                range(num_batches),
                desc=f"Epoch {epoch+1}/{num_epochs}",
            )

            for batch_idx in pbar:
                start = batch_idx * batch_size
                end = start + batch_size
                batch = x_train_shuffled[start:end]

                key, train_key = jax.random.split(key)
                model, opt_state, loss = train_step(model, opt_state, train_key, batch)

                # Log individual batch losses
                wandb.log({
                    "training/batch_loss": float(loss),
                    "training/epoch": epoch,
                    "training/batch": batch_idx,
                    "training/progress": (epoch * num_batches + batch_idx) / (num_epochs * num_batches)
                })

                epoch_losses.append(float(loss))
                pbar.set_postfix({"loss": f"{float(loss):.4f}"})

            # Log epoch-level metrics and images
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            # Compute validation loss
            key, val_key = jax.random.split(key)
            val_loss = compute_validation_loss(model, val_key, x_test)

            # Generate reconstructions and samples
            key, vis_key = jax.random.split(key)
            vis_batch = x_train_shuffled[:num_vis_samples]
            recons = generate_reconstruction_images(model, vis_key, vis_batch)

            # Log all epoch-level metrics and images together
            wandb.log({
                "epoch/train_loss": avg_epoch_loss,
                "epoch/val_loss": float(val_loss),
                "epoch": epoch,
                "reconstructions (left: original, right: reconstruction)": [
                    wandb.Image(
                        np.hstack([
                            np.array(orig.squeeze()),
                            np.ones((28, 4)),
                            np.array(recon.squeeze())
                        ])
                    )
                    for orig, recon in zip(vis_batch, recons)
                ],
                "generated": [
                    wandb.Image(np.array(img.squeeze()))
                    for img in model.generate_samples(key, num_samples=num_vis_samples)
                ],
            })

            checkpoint_path = run_dir / f"vae_checkpoint_epoch_{epoch+1}.eqx"
            model.save(checkpoint_path)

            artifact = wandb.Artifact(
                name=f"model-epoch-{epoch+1}",
                type="model",
                description=f"Model checkpoint at epoch {epoch+1}"
            )
            artifact.add_file(str(checkpoint_path))
            run.log_artifact(artifact)
            # Ensure checkpoint is uploaded even if run crashes
            wandb.save(str(checkpoint_path))

        # Save final model
        final_path = run_dir / "vae_final.eqx"
        model.save(final_path)
        artifact = wandb.Artifact(
            name="model-final",
            type="model",
            description="Final trained model"
        )
        artifact.add_file(str(final_path))
        run.log_artifact(artifact)

    finally:
        wandb.finish()


@jax.jit
def generate_reconstruction_images(model, key, batch):
    key, recon_key = jax.random.split(key)
    recons, _, _ = jax.vmap(lambda x: model(recon_key, x))(batch)
    return recons

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
