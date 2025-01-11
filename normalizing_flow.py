"""
normalizing_flow.py
A simple normalizing flow implementation focused on image generation.
Uses RealNVP-style coupling layers but simplified for MNIST.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

class ConvNet(eqx.Module):
    """Simple ConvNet for the coupling transforms"""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    final: eqx.nn.Linear

    def __init__(self, in_channels: int, out_channels: int, key: PRNGKeyArray):
        keys = jax.random.split(key, 4)
        min_channels = max(in_channels, 1)
        self.conv1 = eqx.nn.Conv2d(min_channels, 32, 3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(32, 32, 3, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(32, 32, 3, padding=1, key=keys[2])
        self.final = eqx.nn.Linear(32 * 28 * 28, 28 * 28 * 2, key=keys[3])

    def __call__(self, x):
        # Single example, shape (C,H,W)
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        # Preserve spatial information
        x = einops.rearrange(x, "c h w -> (c h w)")
        x = self.final(x)
        scale, shift = jnp.split(x, 2)
        scale = scale.reshape(28, 28)
        shift = shift.reshape(28, 28)
        scale = 2 * jnp.tanh(scale/2)
        return scale, shift

class CouplingLayer(eqx.Module):
    net: ConvNet
    mask: Array

    def __init__(self, channels: int, key: PRNGKeyArray, layer_id: int):
        net_key, mask_key = jax.random.split(key)
        # Alternate between vertical and horizontal splits
        if layer_id % 2 == 0:
            self.mask = jnp.concatenate([
                jnp.ones((28, 14)),
                jnp.zeros((28, 14))
            ], axis=1)
        else:
            self.mask = jnp.concatenate([
                jnp.ones((14, 28)),
                jnp.zeros((14, 28))
            ], axis=0)
        self.net = ConvNet(channels, channels, net_key)

    def forward(self, x):
        # Single example, shape (C,H,W)
        x1 = x * self.mask[None, :, :]  # Keep
        x2 = x * (1 - self.mask[None, :, :])  # Transform

        scale, shift = self.net(x1)

        # Apply transformation
        y2 = x2 * jnp.exp(scale[None, :, :]) + shift[None, :, :]
        y = x1 + y2

        log_det = jnp.sum(scale * (1 - self.mask))
        return y, log_det

    def inverse(self, y):
        # Single example
        y1 = y * self.mask[None, :, :]
        y2 = y * (1 - self.mask[None, :, :])

        scale, shift = self.net(y1)

        x2 = (y2 - shift[None, :, :]) * jnp.exp(-scale[None, :, :])
        x = y1 + x2
        return x

class NormalizingFlow(eqx.Module):
    layers: tuple[CouplingLayer, ...]

    def __init__(self, n_layers: int, channels: int, key: PRNGKeyArray):
        keys = jax.random.split(key, n_layers)
        self.layers = tuple(
            CouplingLayer(channels, k, i) for i, k in enumerate(keys)
        )

    def _forward(self, x):
        """Single example forward"""
        total_log_det = 0.
        for layer in self.layers:
            x, log_det = layer.forward(x)
            total_log_det += log_det
        return x, total_log_det

    def _inverse(self, z):
        """Single example inverse"""
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def forward(self, x):
        """Batch forward"""
        return jax.vmap(self._forward)(x)

    def inverse(self, z):
        """Batch inverse"""
        return jax.vmap(self._inverse)(z)

    def _log_prob(self, x):
        """Single example log prob"""
        z, log_det = self._forward(x)
        log_prob = -0.5 * jnp.sum(z**2)
        log_prob -= 0.5 * jnp.prod(jnp.array(z.shape)) * jnp.log(2 * jnp.pi)
        return log_prob + log_det

    def log_prob(self, x):
        """Batch log prob"""
        return jax.vmap(self._log_prob)(x)

    def sample(self, key: PRNGKeyArray, shape):
        z = jax.random.normal(key, shape)
        return self.inverse(z)

    def save(self, filename):
        with open(filename, "wb") as f:
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return eqx.tree_deserialise_leaves(f, cls(
                n_layers=8, channels=1,
                key=jax.random.PRNGKey(0)
            ))
