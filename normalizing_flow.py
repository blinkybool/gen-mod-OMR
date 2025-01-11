"""
normalizing_flow.py
A simple normalizing flow implementation focused on image generation.
Uses RealNVP-style coupling layers but simplified for MNIST.
"""
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class ConvNet(eqx.Module):
    """Transform network for coupling layer"""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    final: eqx.nn.Linear

    def __init__(self, in_channels: int, key: PRNGKeyArray):
        keys = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(in_channels, 32, 3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(32, 32, 3, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(32, 32, 3, padding=1, key=keys[2])
        # Output scale and shift for each pixel
        self.final = eqx.nn.Linear(32 * 28 * 28, 28 * 28 * 2, key=keys[3])

    def __call__(self, x: Array) -> tuple[Array, Array]:
        """
        Args:
            x: Array of shape (C,H,W)
        Returns:
            scale: Array of shape (H,W)
            shift: Array of shape (H,W)
        """
        x = jax.nn.relu(self.conv1(x))  # (32,H,W)
        x = jax.nn.relu(self.conv2(x))  # (32,H,W)
        x = jax.nn.relu(self.conv3(x))  # (32,H,W)
        x = einops.rearrange(x, "c h w -> (c h w)")
        x = self.final(x)  # (2*H*W,)
        scale, shift = jnp.split(x, 2)
        scale = scale.reshape(28, 28)
        shift = shift.reshape(28, 28)
        scale = 2 * jnp.tanh(scale/2)  # Constrain scale
        return scale, shift

class CouplingLayer(eqx.Module):
    """RealNVP-style coupling layer"""
    net: ConvNet
    mask: Array

    def __init__(self, channels: int, layer_id: int, key: PRNGKeyArray):
        net_key, _ = jax.random.split(key)
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
        self.net = ConvNet(channels, net_key)

    def forward(self, x: Array) -> tuple[Array, Array]:
        """
        Args:
            x: Array of shape (C,H,W)
        Returns:
            y: Array of shape (C,H,W)
            log_det: scalar
        """
        x1 = x * self.mask[None, :, :]  # Fixed part
        x2 = x * (1 - self.mask[None, :, :])  # Transformed part

        scale, shift = self.net(x1)
        y2 = x2 * jnp.exp(scale[None, :, :]) + shift[None, :, :]
        y = x1 + y2

        log_det = jnp.sum(scale * (1 - self.mask))
        return y, log_det

    def inverse(self, y: Array) -> Array:
        """
        Args:
            y: Array of shape (C,H,W)
        Returns:
            x: Array of shape (C,H,W)
        """
        y1 = y * self.mask[None, :, :]
        y2 = y * (1 - self.mask[None, :, :])

        scale, shift = self.net(y1)
        x2 = (y2 - shift[None, :, :]) * jnp.exp(-scale[None, :, :])
        x = y1 + x2
        return x

class NormalizingFlow(eqx.Module):
    """Chain of coupling layers"""
    layers: tuple[CouplingLayer, ...]

    def __init__(self, n_layers: int, channels: int, key: PRNGKeyArray):
        keys = jax.random.split(key, n_layers)
        self.layers = tuple(
            CouplingLayer(channels, i, k) for i, k in enumerate(keys)
        )

    def forward(self, x: Array) -> tuple[Array, Array]:
        """
        Args:
            x: Array of shape (C,H,W)
        Returns:
            z: Array of shape (C,H,W)
            log_det: scalar
        """
        total_log_det = 0.
        for layer in self.layers:
            x, log_det = layer.forward(x)
            total_log_det += log_det
        return x, total_log_det

    def inverse(self, z: Array) -> Array:
        """
        Args:
            z: Array of shape (C,H,W)
        Returns:
            x: Array of shape (C,H,W)
        """
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def log_prob(self, x: Array) -> Array:
        """
        Args:
            x: Array of shape (C,H,W)
        Returns:
            log_prob: scalar
        """
        z, log_det = self.forward(x)
        log_prob = -0.5 * jnp.sum(z**2)
        log_prob -= 0.5 * jnp.prod(jnp.array(z.shape)) * jnp.log(2 * jnp.pi)
        return log_prob + log_det

    def sample(self, key: PRNGKeyArray, shape: tuple) -> Array:
        """
        Args:
            key: PRNGKey
            shape: tuple (C,H,W)
        Returns:
            x: Array of shape (C,H,W)
        """
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
