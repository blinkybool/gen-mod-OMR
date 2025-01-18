import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple
import json


def concat_elu(x: Float[Array, "c h w"]) -> Float[Array, "2c h w"]:
    """Activation that applies ELU in both directions and concatenates results"""
    return jnp.concatenate([jax.nn.elu(x), jax.nn.elu(-x)], axis=0)

class GatedConv(eqx.Module):
    """Two-layer convolutional ResNet block with input gate"""
    val_conv: eqx.nn.Conv2d
    gate_conv: eqx.nn.Conv2d

    def __init__(self, c_in: int, c_hidden: int, key: PRNGKeyArray):
        keys = jax.random.split(key, 2)
        self.val_conv = eqx.nn.Conv2d(2*c_in, c_hidden, kernel_size=3, padding=1, key=keys[0])
        self.gate_conv = eqx.nn.Conv2d(2*c_hidden, 2*c_in, kernel_size=1, key=keys[1])

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        # First conv with concat_elu
        h = concat_elu(x)
        h = self.val_conv(h)

        # Second conv with concat_elu and gating
        h = concat_elu(h)
        h = self.gate_conv(h)
        val, gate = jnp.split(h, 2, axis=0)

        return x + val * jax.nn.sigmoid(gate)

class GatedConvNet(eqx.Module):
    """Full conv net with gated activations"""
    init_conv: eqx.nn.Conv2d
    gated_blocks: Tuple[GatedConv, ...]
    norms: Tuple[eqx.nn.LayerNorm, ...]
    final_conv: eqx.nn.Conv2d

    def __init__(self, c_hidden: int, c_out: int, num_layers: int, key: PRNGKeyArray):
        keys = jax.random.split(key, num_layers + 2)

        # Initial conv
        self.init_conv = eqx.nn.Conv2d(1, c_hidden, kernel_size=3, padding=1, key=keys[0])

        # Gated blocks with layer norm
        self.gated_blocks = tuple(
            GatedConv(c_hidden, c_hidden, key=keys[i+1])
            for i in range(num_layers)
        )
        self.norms = tuple(
            eqx.nn.LayerNorm((c_hidden, 28, 28))  # Changed to match full shape
            for _ in range(num_layers)
        )

        self.final_conv = eqx.nn.Conv2d(
            2*c_hidden, c_out, kernel_size=3, padding=1,
            key=keys[-1]
        )

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c_out h w"]:
        h = self.init_conv(x)

        for block, norm in zip(self.gated_blocks, self.norms):
            h = block(h)
            h = norm(h)

        h = concat_elu(h)
        return self.final_conv(h)


class CouplingLayer(eqx.Module):
    """RealNVP-style coupling layer using shift-then-scale transformation"""
    net: GatedConvNet
    mask: Float[Array, "1 h w"]  # Binary mask where 0 denotes transform, 1 denotes keep
    scaling_factor: Float[Array, "1"]  # Learned factor for scale stabilization

    def __init__(self, mask: Float[Array, "1 h w"], key: PRNGKeyArray):
        self.mask = mask
        self.net = GatedConvNet(c_hidden=32, c_out=2, num_layers=3, key=key)
        self.scaling_factor = jnp.zeros(1)  # Initialize at zero like tutorial

    def get_transform_params(self, x: Float[Array, "1 h w"]) -> Tuple[Float[Array, "1 h w"], Float[Array, "1 h w"]]:
        """Compute scale and shift parameters from masked input"""
        z_in = x * self.mask
        net_out = self.net(z_in)
        scale, shift = jnp.split(net_out, 2, axis=0)

        # Stabilize scaling output
        scale = jnp.tanh(scale / jnp.exp(self.scaling_factor)) * jnp.exp(self.scaling_factor)

        # Mask outputs (only transform the unmasked parts)
        scale = scale * (1 - self.mask)
        shift = shift * (1 - self.mask)

        return scale, shift

    def forward(self, x: Float[Array, "1 h w"]) -> Tuple[Float[Array, "1 h w"], Float[Array, ""]]:
        scale, shift = self.get_transform_params(x)
        x = (x + shift) * jnp.exp(scale)
        ldj = jnp.sum(scale)
        return x, ldj

    def inverse(self, z: Float[Array, "1 h w"]) -> Tuple[Float[Array, "1 h w"], Float[Array, ""]]:
        scale, shift = self.get_transform_params(z)
        x = (z * jnp.exp(-scale)) - shift
        ldj = -jnp.sum(scale)
        return x, ldj


ALPHA = 1e-5  # Numerical stability constant
QUANTS = 256  # Number of pixel values [0,255]

def dequant(x: Float[Array, "c h w"], rng: PRNGKeyArray) -> Tuple[Float[Array, "c h w"], Float[Array, ""], PRNGKeyArray]:
    """Convert discrete values to continuous by adding noise and applying logit transform"""
    # First dequant
    x = x.astype(jnp.float32)
    rng, uniform_rng = jax.random.split(rng)
    x = x + jax.random.uniform(uniform_rng, x.shape)
    x = x / QUANTS

    c, h, w = x.shape
    n_dims = c * h * w  # Multiply CHW dimensions
    ldj = -jnp.log(QUANTS) * n_dims

    # Then sigmoid inverse
    x = x * (1 - ALPHA) + 0.5 * ALPHA
    ldj += jnp.log(1 - ALPHA) * n_dims
    ldj += (-jnp.log(x) - jnp.log(1-x)).sum()
    x = jnp.log(x) - jnp.log(1-x)

    return x, ldj, rng

def quantize(z: Float[Array, "c h w"]) -> Tuple[Float[Array, "c h w"], Float[Array, ""]]:
    """Convert continuous values back to discrete pixel space"""
    # First sigmoid
    ldj = (-z-2*jax.nn.softplus(-z)).sum()
    x = jax.nn.sigmoid(z)

    c, h, w = x.shape
    n_dims = c * h * w  # Multiply CHW dimensions
    ldj -= jnp.log(1 - ALPHA) * n_dims
    x = (x - 0.5 * ALPHA) / (1 - ALPHA)

    # Then quantize
    x = x * QUANTS
    ldj += jnp.log(QUANTS) * n_dims
    x = jnp.floor(x)
    x = jnp.clip(x, 0, QUANTS-1).astype(jnp.int32)

    return x, ldj


class NormalizingFlow(eqx.Module):
    """Main flow combining dequantization and coupling layers"""
    coupling_layers: Tuple[CouplingLayer, ...]

    def __init__(self, n_layers: int, key: PRNGKeyArray):
        keys = jax.random.split(key, n_layers)

        # Create checkerboard masks, alternating True/False
        h, w = 28, 28  # MNIST size
        self.coupling_layers = tuple(
            CouplingLayer(
                mask=create_checkerboard_mask(h, w, i % 2 == 0),
                key=keys[i]
            )
            for i in range(n_layers)
        )

    def forward(self, x: Float[Array, "c h w"], rng: PRNGKeyArray) -> Tuple[Float[Array, "c h w"], Float[Array, ""], PRNGKeyArray]:
        """Transform input to latent space and compute log det jacobian"""
        # Dequantize (only step needing RNG)
        z, ldj, rng = dequant(x, rng)

        # Apply coupling layers
        for layer in self.coupling_layers:
            z, layer_ldj = layer.forward(z)
            ldj += layer_ldj

        return z, ldj, rng

    def inverse(self, z: Float[Array, "c h w"]) -> Tuple[Float[Array, "c h w"], Float[Array, ""]]:
        """Transform from latent space back to input space and compute log det jacobian"""
        x = z
        ldj = 0.0

        # Apply coupling layers in reverse
        for layer in reversed(self.coupling_layers):
            x, layer_ldj = layer.inverse(x)
            ldj += layer_ldj

        # Finally apply dequantization inverse
        x, quantize_ldj = quantize(x)
        ldj += quantize_ldj

        return x, ldj

    def log_prob(self, x: Float[Array, "c h w"], rng: PRNGKeyArray) -> Float[Array, ""]:
        """Compute log probability of input"""
        # Ensure x has shape (1,h,w)
        if len(x.shape) != 3:
            x = x.reshape(1, 28, 28)
        z, ldj, _ = self.forward(x, rng)
        c, h, w = z.shape
        prior_logprob = -0.5 * jnp.sum(z**2) - 0.5 * c * h * w * jnp.log(2 * jnp.pi)
        return prior_logprob + ldj

    def sample(self, rng: PRNGKeyArray) -> Tuple[Float[Array, "c h w"], Float[Array, ""]]:
        """Generate a sample by drawing from standard normal and inverting"""
        z_rng, rng = jax.random.split(rng)
        z = jax.random.normal(z_rng, (1, 28, 28))
        x, ldj = self.inverse(z)
        return x, ldj

    def save(self, filename: str):
        """Save model parameters"""
        with open(filename, "wb") as f:
            # Save hyperparameters
            hyperparams = {
                "n_layers": len(self.coupling_layers),
            }
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())

            # Save model weights
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(cls, filename: str):
        """Load model from file"""
        with open(filename, "rb") as f:
            # Load hyperparameters
            hyperparams = json.loads(f.readline().decode())

            # Create model with same architecture
            model = cls(
                n_layers=hyperparams["n_layers"],
                key=jax.random.PRNGKey(0)  # Key doesn't matter for loading
            )

            # Load weights into model
            return eqx.tree_deserialise_leaves(f, model)


def create_checkerboard_mask(h: int, w: int, invert: bool = False) -> Float[Array, "1 h w"]:
    """Create a checkerboard mask of zeros and ones"""
    x, y = jnp.arange(h), jnp.arange(w)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    mask = jnp.fmod(xx + yy, 2)
    mask = mask.astype(jnp.float32).reshape(1, h, w)
    if invert:
        mask = 1 - mask
    return mask
