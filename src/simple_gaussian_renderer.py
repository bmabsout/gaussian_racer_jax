import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
from functools import partial
from dataclasses import dataclass

@dataclass(frozen=True)
class ViewRect:
    """Represents a viewport rectangle in world space."""
    center: jnp.ndarray  # Center position [x, y]
    size: jnp.ndarray    # Width and height
    pixels: tuple[int, int]  # Output resolution (width, height)

@jax.jit
def compute_gaussian(
    coords: jnp.ndarray,
    mean: jnp.ndarray,
    std: float,
    amplitude: float = 1.0
) -> jnp.ndarray:
    """Compute a single 2D Gaussian."""
    diff = coords - mean
    exponent = -0.5 * jnp.sum(diff**2, axis=-1) / (std**2)
    return amplitude * jnp.exp(exponent)

@partial(jax.jit, static_argnums=(5, 6))
def render_gaussians(
    points: jnp.ndarray,      # World space points
    intensities: jnp.ndarray,
    stds: jnp.ndarray,
    center: jnp.ndarray,      # View center
    size: jnp.ndarray,        # View size
    width: int,               # Output width
    height: int               # Output height
) -> jnp.ndarray:
    """Render multiple 2D Gaussians into a viewport rectangle."""
    # Create pixel coordinate grid
    x, y = jnp.meshgrid(
        jnp.linspace(-size[0]/2, size[0]/2, width),
        jnp.linspace(-size[1]/2, size[1]/2, height),
        indexing='xy'
    )
    # Transform to world space
    x = x + center[0]
    y = y + center[1]
    # Stack coordinates in the same order as points [x, y]
    coords = jnp.stack([x, y], axis=-1).transpose(1, 0, 2)  # Added transpose
    
    # Compute all gaussians using vmap
    vectorized_gaussian = vmap(
        lambda p, s, a: compute_gaussian(coords, p, s, a)
    )
    gaussians = vectorized_gaussian(points, stds, intensities)
    
    # Sum all gaussians
    image = jnp.sum(gaussians, axis=0)
    return jnp.clip(image/2.0, 0.0, 1.0)

def create_random_points(n_points: int = 2000, spread: float = 500.0):
    """Create random gaussian points in world space."""
    key = jax.random.PRNGKey(0)
    points = jax.random.normal(key, shape=(n_points, 2)) * spread
    intensities = jnp.ones(n_points) * 0.5
    stds = jnp.ones(n_points) * 20.0
    return points, intensities, stds
