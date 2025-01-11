from pathlib import Path

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

import wandb
from normalizing_flow import NormalizingFlow


def get_platform():
    return jax.default_backend()

def configure_platform():
    platform = get_platform()
    if platform == "tpu":
        print("Using jax TPU")
        jax.config.update("jax_enable_x64", False)
    else:
        pass

def main(
    n_layers: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    num_epochs: int = 50,
    num_vis_samples: int = 10,
    seed: int = 42,
    output_folder: Path = Path("runs/nf"),
    wandb_project: str = "nf-mnist",
    wandb_entity: str | None = None,
):
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
        model = NormalizingFlow(n_layers=n_layers, channels=1, key=model_key)
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        @jax.jit
        def compute_batch_loss(model, batch):
            return -jnp.mean(model.log_prob(batch))

        @jax.jit
        def train_step(model, opt_state, batch):
            loss_fn = lambda model: compute_batch_loss(model, batch)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
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

                model, opt_state, loss = train_step(model, opt_state, batch)

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
            val_loss = compute_batch_loss(model, x_test)

            # Generate samples
            key, sample_key = jax.random.split(key)
            samples = model.sample(
                sample_key,
                shape=(num_vis_samples, 1, 28, 28)
            )

            wandb.log({
                "epoch/train_loss": avg_epoch_loss,
                "epoch/val_loss": float(val_loss),
                "epoch": epoch,
                "samples": [
                    wandb.Image(np.array(img.squeeze()))
                    for img in samples
                ],
            })

            checkpoint_path = run_dir / f"nf_checkpoint_epoch_{epoch+1}.eqx"
            model.save(checkpoint_path)

            artifact = wandb.Artifact(
                name=f"model-epoch-{epoch+1}",
                type="model",
                description=f"Model checkpoint at epoch {epoch+1}"
            )
            artifact.add_file(str(checkpoint_path))
            run.log_artifact(artifact)
            wandb.save(str(checkpoint_path))

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
