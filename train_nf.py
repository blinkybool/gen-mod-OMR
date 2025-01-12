import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from pathlib import Path
import tempfile

from normalizing_flow import NormalizingFlow

def get_platform():
    """Returns the current hardware platform (cpu, gpu, or tpu)"""
    return jax.default_backend()

def configure_platform():
    """Configure JAX for the current platform"""
    platform = get_platform()
    if platform == "tpu":
        print("Using jax TPU")
        jax.config.update("jax_enable_x64", False)
    else:
        print(f"Using jax {platform}")

def generate_samples(model, key, num_samples):
    """Generate samples using vmap to handle batching"""
    keys = jax.random.split(key, num_samples)
    samples, _ = jax.vmap(model.sample)(keys)  # Ignore returned RNGs
    return samples

def main(
    n_layers: int = 12,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    num_epochs: int = 100,
    num_vis_samples: int = 10,
    seed: int = 42,
    output_folder: Path = Path("runs/nf"),
    wandb_project: str = "normalizing-flow-mnist",
    wandb_entity: str | None = None,
):
    """Train a new normalizing flow model"""

    configure_platform()
    output_folder.mkdir(parents=True, exist_ok=True)
    Path("wandb").mkdir(exist_ok=True)

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

    try:
        # Create run-specific output directory
        run_dir = output_folder / run.id
        run_dir.mkdir(parents=True, exist_ok=True)

        key = jax.random.key(seed)

        print("Loading mnist.npz...")
        with np.load("mnist.npz") as data:
            x_train = jnp.array(data["x_train"].astype(np.int32))
            x_test = jnp.array(data["x_test"].astype(np.int32))
        # Convert to CHW format
        x_train = einops.rearrange(x_train, "b h w -> b 1 h w")
        x_test = einops.rearrange(x_test, "b h w -> b 1 h w")

        num_batches = len(x_train) // batch_size
        print(f"Training on {len(x_train)} images with {num_batches} batches per epoch")

        print("Initializing model/optimizer...")
        key, model_key = jax.random.split(key)
        model = NormalizingFlow(n_layers=n_layers, key=model_key)
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        def log_prob_batch(model, x, rng):
            """Compute log prob for a single example"""
            # Remove batch dimension for individual processing
            x = einops.rearrange(x, "1 c h w -> c h w")
            return model.log_prob(x, rng)

        @jax.jit
        def compute_batch_loss(model, batch, rng):
            """Compute loss for a batch using scan instead of vmap"""
            keys = jax.random.split(rng, batch.shape[0])
            def body_fun(carry, args):
                x, key = args
                log_prob, _ = log_prob_batch(model, x[None], key)
                return carry, log_prob
            _, log_probs = jax.lax.scan(body_fun, None, (batch, keys))
            return -jnp.mean(log_probs), rng

        @jax.jit
        def train_step(model, opt_state, batch, rng):
            """Single training step"""
            (loss, rng), grads = eqx.filter_value_and_grad(compute_batch_loss, has_aux=True)(model, batch, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, rng

        print("Beginning training...")

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

                model, opt_state, loss, key = train_step(model, opt_state, batch, key)

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
            key, val_key = jax.random.split(key)
            val_loss, _ = compute_batch_loss(model, x_test, val_key)

            # Generate samples
            key, sample_key = jax.random.split(key)
            samples = generate_samples(model, sample_key, num_vis_samples)

            wandb.log({
                "epoch/train_loss": avg_epoch_loss,
                "epoch/val_loss": val_loss,
                "epoch": epoch,
                "samples": [
                    wandb.Image(np.array(img.squeeze()))
                    for img in samples
                ],
            })

            # Save checkpoint every 5 epochs directly to wandb (no local save)
            if (epoch + 1) % 5 == 0:
                with tempfile.NamedTemporaryFile() as tmp:
                    model.save(tmp.name)
                    artifact = wandb.Artifact(
                        name=f"model-epoch-{epoch+1}",
                        type="model",
                        description=f"Model checkpoint at epoch {epoch+1}"
                    )
                    artifact.add_file(tmp.name)
                    run.log_artifact(artifact)

        # Save final model
        final_path = run_dir / "nf_final.eqx"
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

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
