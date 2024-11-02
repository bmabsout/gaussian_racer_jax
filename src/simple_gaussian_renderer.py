import jax
import jax.numpy as jnp
from typing import NamedTuple

class Gaussians(NamedTuple):
    """Represents a collection of 2D Gaussians with parallel arrays."""
    pos: jnp.ndarray    # shape: (n, 2) for positions
    std: jnp.ndarray    # shape: (n,)
    intensity: jnp.ndarray  # shape: (n,)
    
    @staticmethod
    def compose(*gaussians: 'Gaussians') -> 'Gaussians':
        """Combine multiple Gaussians objects."""
        if not gaussians:
            return Gaussians(
                pos=jnp.zeros((0, 2)),
                std=jnp.zeros(0),
                intensity=jnp.zeros(0)
            )
        return Gaussians(
            pos=jnp.concatenate([g.pos for g in gaussians]),
            std=jnp.concatenate([g.std for g in gaussians]),
            intensity=jnp.concatenate([g.intensity for g in gaussians])
        )
    def render_at_positions(
        self,
        positions: jnp.ndarray,  # shape: (n_pixels, 2)
    ) -> jnp.ndarray:
        """Compute gaussian values for each position."""
        x_diff = positions[:, 0, None] - self.pos[:, 0]
        y_diff = positions[:, 1, None] - self.pos[:, 1]
        sq_distances = x_diff**2 + y_diff**2
        values = self.intensity * jnp.exp(-0.5 * sq_distances / self.std**2)
        return jnp.sum(values, axis=1)
