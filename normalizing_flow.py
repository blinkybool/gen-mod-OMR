import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple
import json


class GatedConvNet(eqx.Module):
    """CNN with gated activations and residual connections"""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    final: eqx.nn.Conv2d

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, key: PRNGKeyArray):
        keys = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(in_channels, hidden_channels, 3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, key=keys[2])
        self.norm = eqx.nn.GroupNorm(groups=1, channels=hidden_channels)
        self.final = eqx.nn.Conv2d(hidden_channels, out_channels, 3, padding=1, key=keys[3])

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        h = jax.nn.relu(self.conv1(x))
        h = h + jax.nn.relu(self.conv2(h))  # Residual
        h = h + jax.nn.relu(self.conv3(h))  # Residual
        h = self.norm(h)
        return self.final(h)


class CouplingLayer(eqx.Module):
    """RealNVP-style coupling layer"""
    net: GatedConvNet
    mask: Float[Array, "1 h w"]

    def __init__(self, mask: Float[Array, "1 h w"], key: PRNGKeyArray):
        self.mask = mask
        self.net = GatedConvNet(in_channels=1, hidden_channels=32, out_channels=2, key=key)

    def forward(self, x: Float[Array, "1 h w"]) -> Tuple[Float[Array, "1 h w"], Float[Array, ""]]:
        # Get transformation parameters
        masked_x = x * self.mask
        net_out = self.net(masked_x)

        # Split into scale and shift (first and second channel)
        scale = 2 * jnp.tanh(net_out[0:1] / 2)  # Constrain scale
        shift = net_out[1:2]

        # Transform second part
        x = x * jnp.exp(scale * (1-self.mask)) + shift * (1-self.mask)
        ldj = jnp.sum(scale * (1-self.mask))
        return x, ldj

    def inverse(self, z: Float[Array, "1 h w"]) -> Float[Array, "1 h w"]:
        # Get transformation parameters
        masked_z = z * self.mask
        net_out = self.net(masked_z)
        scale = 2 * jnp.tanh(net_out[0:1] / 2)
        shift = net_out[1:2]

        # Invert transformation
        x = (z - shift * (1-self.mask)) * jnp.exp(-scale * (1-self.mask))
        return x


class Dequantization(eqx.Module):
    """Dequantization layer for converting discrete to continuous"""
    def __call__(self, x: Float[Array, "c h w"], rng: PRNGKeyArray) -> Tuple[Float[Array, "c h w"], Float[Array, ""], PRNGKeyArray]:
        # Add uniform noise and rescale to [0,1]
        x = x.astype(jnp.float32)
        noise_rng, rng = jax.random.split(rng)
        x = x + jax.random.uniform(noise_rng, x.shape)
        x = x / 256.0

        # Transform to have better numerical properties
        x = x * 2 - 1  # Scale to [-1,1]
        ldj = -jnp.log(256.0) * jnp.prod(jnp.array(x.shape))
        return x, ldj, rng


class NormalizingFlow(eqx.Module):
    """Main flow combining dequantization and coupling layers"""
    dequant: Dequantization
    coupling_layers: Tuple[CouplingLayer, ...]

    def __init__(self, n_layers: int, key: PRNGKeyArray):
        keys = jax.random.split(key, n_layers)

        # Create checkerboard masks, alternating True/False
        h, w = 28, 28  # MNIST size
        self.dequant = Dequantization()
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
        z, ldj, rng = self.dequant(x, rng)

        # Apply coupling layers
        for layer in self.coupling_layers:
            z, layer_ldj = layer.forward(z)
            ldj += layer_ldj

        return z, ldj, rng

    def inverse(self, z: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        """Transform from latent space back to input space"""
        x = z
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x: Float[Array, "c h w"], rng: PRNGKeyArray) -> Tuple[Float[Array, ""], PRNGKeyArray]:
        """Compute log probability of input"""
        z, ldj, rng = self.forward(x, rng)
        prior_logprob = -0.5 * jnp.sum(z**2) - 0.5 * jnp.prod(jnp.array(z.shape)) * jnp.log(2 * jnp.pi)
        return prior_logprob + ldj, rng

    def sample(self, rng: PRNGKeyArray) -> Tuple[Float[Array, "c h w"], PRNGKeyArray]:
        """Generate a sample by drawing from standard normal and inverting"""
        z_rng, rng = jax.random.split(rng)
        z = jax.random.normal(z_rng, (1, 28, 28))
        x = self.inverse(z)
        return x, rng

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
