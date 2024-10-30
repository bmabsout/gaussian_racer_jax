import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple
import numpy as np
import matplotlib.cm as cm

# Pre-compute colormap
COLORMAP = (cm.get_cmap('inferno')(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

def apply_colormap(image: np.ndarray) -> np.ndarray:
    """Apply the inferno colormap to an image."""
    return COLORMAP[(image * 255).astype(np.uint8)]

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

def create_random_gaussians(n_points: int = 2000, spread: float = 500.0) -> Gaussians:
    """Create random gaussian points in world space."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    
    return Gaussians(
        pos=jax.random.normal(k1, shape=(n_points, 2)) * spread,
        std=jnp.exp(jax.random.normal(k2, shape=(n_points,))) * 10.0,
        intensity=jax.random.uniform(k3, shape=(n_points,), minval=0.2, maxval=0.5)
    )

def render_gaussians_at_positions(
    positions: jnp.ndarray,  # shape: (n_pixels, 2)
    gaussians: Gaussians
) -> jnp.ndarray:
    """Compute gaussian values for each position."""
    # Compute distances between pixels and gaussians
    diff = positions[None, :, :] - gaussians.pos[:, None, :]  # shape: (n_gaussians, n_pixels, 2)
    sq_distances = jnp.sum(diff**2, axis=-1)  # shape: (n_gaussians, n_pixels)
    
    # Compute gaussian values
    values = gaussians.intensity[:, None] * jnp.exp(-0.5 * sq_distances / (gaussians.std[:, None]**2))
    
    # Sum contributions from all gaussians
    return jnp.sum(values, axis=0)  # shape: (n_pixels,)

@partial(jax.jit, static_argnums=(3, 4))
def render_view(
    gaussians: Gaussians,
    view_center: jnp.ndarray,
    view_size: jnp.ndarray,
    width: int,
    height: int
) -> jnp.ndarray:
    """Render gaussians for a given view (all in world coordinates)."""
    # Create pixel grid in world space
    x = jnp.linspace(-view_size[0]/2, view_size[0]/2, width)
    y = jnp.linspace(-view_size[1]/2, view_size[1]/2, height)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    positions = jnp.stack([X + view_center[0], Y + view_center[1]], axis=-1)
    
    # Compute gaussian values for all positions
    values = render_gaussians_at_positions(positions.reshape(-1, 2), gaussians)
    
    # Reshape to image and normalize
    image = values.reshape(width, height)
    return jnp.clip(image/2.0, 0.0, 1.0)
