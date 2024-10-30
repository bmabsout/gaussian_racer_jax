import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple
from src.pygame_utils import WindowConfig, run_renderer
from src.gaussian_utils import apply_colormap
import jax.random

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
        std=jnp.exp(jax.random.normal(k2, shape=(n_points,))) * 10.0,  # Log-normal distribution for positive stds
        intensity=jax.random.uniform(k3, shape=(n_points,), minval=0.2, maxval=0.5)
    )

@partial(jax.jit, static_argnums=(3, 4))
def render_view(
    gaussians: Gaussians,
    view_center: jnp.ndarray,
    view_size: jnp.ndarray,
    width: int,
    height: int
) -> jnp.ndarray:
    """Render gaussians for a given view (all in world coordinates)."""
    # Create coordinate grid
    x = jnp.linspace(-view_size[0]/2, view_size[0]/2, width)
    y = jnp.linspace(-view_size[1]/2, view_size[1]/2, height)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    # Pre-compute transformed coordinates
    coords = jnp.stack([
        X + view_center[0],
        Y + view_center[1]
    ], axis=-1)
    
    # Vectorized gaussian computation
    diff = coords[None, :, :, :] - gaussians.pos[:, None, None, :]
    sq_distances = jnp.sum(diff**2, axis=-1)
    values = gaussians.intensity[:, None, None] * jnp.exp(
        -0.5 * sq_distances / (gaussians.std[:, None, None]**2)
    )
    
    # Sum and normalize
    image = jnp.sum(values, axis=0)
    return jnp.clip(image/2.0, 0.0, 1.0)

if __name__ == "__main__":
    # Create initial state
    gaussians = create_random_gaussians()
    
    # Create renderer function that matches the expected interface
    def renderer(mouse_world_pos, view_center, view_size, width, height):
        # Convert numpy arrays to jax arrays
        view_center = jnp.array(view_center)
        view_size = jnp.array(view_size)
        
        # Create mouse gaussian and compose with background
        mouse_gaussian = Gaussians(
            pos=jnp.array(mouse_world_pos)[None, :],
            std=jnp.array([20.0]),
            intensity=jnp.array([1.0])
        )
        all_gaussians = Gaussians.compose(gaussians, mouse_gaussian)
        
        image = render_view(all_gaussians, view_center, view_size, width, height)
        return apply_colormap(image)
    
    # Run with pygame
    config = WindowConfig(
        width=1024,
        height=768,
        title="Simple Gaussian Renderer"
    )
    run_renderer(config, renderer)
