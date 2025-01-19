import tempfile
from pathlib import Path
from typing import Tuple

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from jaxtyping import Array, Float

import wandb
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
    """Generate samples using vmap"""
    keys = jax.random.split(key, num_samples)
    samples = jax.vmap(lambda k: model.sample(k)[0])(keys)
    return samples

def bits_per_dim(log_probs: Float[Array, "..."], shape: Tuple[int, ...]) -> Float[Array, "..."]:
    """Convert log probabilities to bits per dimension"""
    return -log_probs * jnp.log2(jnp.e) / (shape[-3] * shape[-2] * shape[-1])

def evaluate_in_batches(model, x_test, batch_size, key):
    num_test_batches = len(x_test) // batch_size + (1 if len(x_test) % batch_size != 0 else 0)
    all_bpds = []

    for i in range(num_test_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(x_test))
        batch = x_test[start_idx:end_idx]

        # Generate keys for this batch
        batch_keys = jax.random.split(key, batch.shape[0])
        key = jax.random.fold_in(key, i)  # Get different keys for next batch

        # Compute log probs for batch
        batch_log_probs = jax.vmap(lambda x, k: model.log_prob(x, k))(batch, batch_keys)
        batch_bpd = bits_per_dim(-batch_log_probs, batch.shape)
        all_bpds.append(batch_bpd)

    # Concatenate all results
    return jnp.concatenate(all_bpds)

def main(
    n_layers: int = 8,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    num_epochs: int = 200,
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
        # Convert to NCHW format
        x_train = einops.rearrange(x_train, "b h w -> b 1 h w")
        x_test = einops.rearrange(x_test, "b h w -> b 1 h w")

        num_batches = len(x_train) // batch_size
        print(f"Training on {len(x_train)} images with {num_batches} batches per epoch")

        print("Initializing model/optimizer...")
        key, model_key = jax.random.split(key)
        model = NormalizingFlow(n_layers=n_layers, key=model_key)

        # Add learning rate schedule and gradient clipping
        lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=num_batches,
            decay_rate=0.99,
            end_value=0.01 * learning_rate
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_schedule)
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        @jax.jit
        def compute_batch_loss(model, batch, rng):
            """Compute bits per dimension loss for a batch"""
            keys = jax.random.split(rng, batch.shape[0])
            log_probs = jax.vmap(lambda x, k: model.log_prob(x, k))(batch, keys)
            return bits_per_dim(-jnp.mean(log_probs), batch.shape), rng

        @jax.jit
        def train_step(model, opt_state, batch, rng):
            """Single training step"""
            (loss, rng), grads = eqx.filter_value_and_grad(
                compute_batch_loss, has_aux=True
            )(model, batch, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, rng

        print("Beginning training...")
        best_val_bpd = float('inf')

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

                model, opt_state, bpd, key = train_step(model, opt_state, batch, key)

                wandb.log({
                    "training/batch_bpd": float(bpd),
                    "training/epoch": epoch,
                    "training/batch": batch_idx,
                    "training/progress": (epoch * num_batches + batch_idx) / (num_epochs * num_batches)
                })

                epoch_losses.append(float(bpd))
                pbar.set_postfix({"bpd": f"{float(bpd):.4f}"})

            # Regular evaluation
            avg_train_bpd = sum(epoch_losses) / len(epoch_losses)
            key, val_key = jax.random.split(key)
            val_bpd = evaluate_in_batches(model, x_test, batch_size, val_key)
            val_bpd_mean = float(val_bpd.mean())

            # Generate samples
            key, sample_key = jax.random.split(key)
            samples = generate_samples(model, sample_key, num_vis_samples)

            wandb.log({
                "epoch/train_bpd": avg_train_bpd,
                "epoch/val_bpd": val_bpd_mean,
                "epoch": epoch,
                "samples": [
                    wandb.Image(np.array(img.squeeze()))
                    for img in samples
                ],
            })

            # Save best model based on validation BPD
            if val_bpd_mean < best_val_bpd:
                best_val_bpd = val_bpd_mean
                best_path = run_dir / "nf_best.eqx"
                model.save(best_path)
                artifact = wandb.Artifact(
                    name="model-best",
                    type="model",
                    description=f"Best model (val_bpd={best_val_bpd:.4f})"
                )
                artifact.add_file(str(best_path))
                run.log_artifact(artifact)

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
