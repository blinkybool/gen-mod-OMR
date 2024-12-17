import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import shutil


from vae import VAE, compute_loss, vis_samples

def main(
    latent_dim: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    num_epochs: int = 50,
    batches_per_visual: int = 20,
    checkpoint_every: int = 1,
    seed: int = 42,
):
    key = jax.random.key(seed)

    print("initialising model...")
    with np.load("mnist.npz") as data:
        x_train = jnp.array(data["x_train"].astype("float32") / 255.0)
    x_train = einops.rearrange(x_train, "b h w -> b 1 h w")

    # Calculate number of complete batches
    num_batches = len(x_train) // batch_size

    print("initialising optimiser...")
    key, model_key = jax.random.split(key)
    model = VAE(latent_dim=latent_dim, key=model_key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @jax.jit
    def train_step(model, opt_state, key, batch):
        batch = batch.astype(jnp.float32)

        def batch_loss(model):
            keys = jax.random.split(key, batch.shape[0])

            def single_loss(x, k):
                return compute_loss(model, k, x)

            losses = jax.vmap(single_loss)(batch, keys)
            return jnp.mean(losses)

        loss, grads = eqx.filter_value_and_grad(batch_loss)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    print("begin training...")

    for epoch in range(num_epochs):
        # Shuffle once per epoch
        key, shuffle_key = jax.random.split(key)
        x_train_shuffled = x_train[
            jax.random.permutation(shuffle_key, len(x_train))
        ]

        pbar = tqdm.tqdm(
            range(num_batches),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        for i in pbar:
            # Get one batch
            start = i * batch_size
            end = start + batch_size
            batch = x_train_shuffled[start:end]

            # Train on batch
            key, train_key = jax.random.split(key)
            model, opt_state, loss = train_step(
                model, opt_state, train_key, batch
            )
            pbar.set_postfix({"loss": float(loss)})

            # Visualize every few batches
            if i % batches_per_visual == 0:
                key, sample_key = jax.random.split(key)
                terminal_width = shutil.get_terminal_size().columns
                max_columns = terminal_width // 30  # Each sample is ~32 chars wide
                num_samples = min(max_columns, 10)  # Cap at 10 samples
                samples = model.generate_samples(sample_key, num_samples=num_samples)
                plot = vis_samples(samples, columns=num_samples)
                print("\nSamples:")
                print(plot)
                print()

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            model.save(
                f"models/vae_checkpoint_epoch_{epoch+1}_latent{latent_dim}_lr{learning_rate}_batch{batch_size}.eqx"
            )
    # Save final model
    model.save(
        f"models/vae_final_latent{latent_dim}_lr{learning_rate}_batch{batch_size}.eqx",
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
