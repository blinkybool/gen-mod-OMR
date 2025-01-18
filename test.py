import jax
import jax.numpy as jnp
import equinox as eqx
from normalizing_flow import NormalizingFlow
import einops

def debug_model():
    print("=== Starting Debug ===")

    # 1. Test model initialization
    print("\n1. Testing model initialization...")
    key = jax.random.PRNGKey(0)
    try:
        model = NormalizingFlow(n_layers=2, key=key)
        print("✓ Model initialization successful")
    except Exception as e:
        print("✗ Model initialization failed:", e)
        return

    # 2. Create test batch
    print("\n2. Creating test batch...")
    batch_size = 2
    try:
        # Create a small batch in NCHW format and convert to float32
        batch = jnp.ones((batch_size, 1, 28, 28), dtype=jnp.int32)
        print(f"✓ Test batch created with shape: {batch.shape}, dtype: {batch.dtype}")
    except Exception as e:
        print("✗ Test batch creation failed:", e)
        return

    # 3. Test single example log_prob
    print("\n3. Testing single example log_prob...")
    try:
        key, subkey = jax.random.split(key)
        single_x = batch[0]  # Take first example
        print(f"Single example shape: {single_x.shape}")
        log_prob = model.log_prob(single_x, subkey)
        print(f"✓ Log prob computation successful: {log_prob}")
    except Exception as e:
        print("✗ Log prob computation failed:", e)
        import traceback
        traceback.print_exc()
        return

    # 4. Test batch processing
    print("\n4. Testing batch processing...")
    try:
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)

        def single_example_loss(x_and_key):
            x, key = x_and_key
            print(f"Shape before reshape: {x.shape}")
            x_single = x.reshape(1, 28, 28)  # Ensure (c,h,w) format
            print(f"Shape after reshape: {x_single.shape}")
            return model.log_prob(x_single, key)

        # Test single example first
        test_x, test_key = batch[0], keys[0]
        print(f"Testing single example processing...")
        single_result = single_example_loss((test_x, test_key))
        print(f"Single example result: {single_result}")

        # Then test vmap
        print(f"Testing vmap over batch...")
        log_probs = jax.vmap(single_example_loss)((batch, keys))
        print(f"✓ Batch processing successful, log_probs shape: {log_probs.shape}")

    except Exception as e:
        print("✗ Batch processing failed:", e)
        import traceback
        traceback.print_exc()
        return

    # 5. Test gradient computation
    print("\n5. Test gradient computation...")
    try:
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)

        def compute_batch_loss(model, batch, keys):
            log_probs = jax.vmap(lambda x, k: model.log_prob(x, k))(batch, keys)
            return -jnp.mean(log_probs)

        value, grads = eqx.filter_value_and_grad(
            lambda m: compute_batch_loss(m, batch, keys)
        )(model)
        print("✓ Gradient computation successful")
        print(f"Loss value: {value}")

        # Print gradient structure
        print("\nGradient structure:")
        jax.tree_map(lambda x: print(f"Shape: {x.shape if hasattr(x, 'shape') else None}, Type: {type(x)}"), grads)

        # Check for None gradients
        def check_grads(g, path=""):
            if isinstance(g, dict):
                for k, v in g.items():
                    check_grads(v, f"{path}.{k}")
            elif isinstance(g, (tuple, list)):
                for i, v in enumerate(g):
                    check_grads(v, f"{path}[{i}]")
            else:
                if g is None:
                    print(f"✗ Found None gradient at {path}")
                    return False
            return True

        all_good = check_grads(grads)
        if all_good:
            print("✓ No None gradients found")

    except Exception as e:
        print("✗ Gradient computation failed:", e)
        import traceback
        traceback.print_exc()
        return

    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_model()
