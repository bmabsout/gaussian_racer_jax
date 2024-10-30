import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from src.pygame_utils import WindowConfig, run_renderer
from src.gaussian_utils import apply_colormap

def create_random_points(n_points: int = 2000, spread: float = 500.0):
    """Create random gaussian points in world space."""
    key = jax.random.PRNGKey(0)
    points = jax.random.normal(key, shape=(n_points, 2)) * spread
    intensities = jnp.ones(n_points) * 0.5
    stds = jnp.ones(n_points) * 20.0
    return points, intensities, stds

@partial(jax.jit, static_argnums=(6, 7))
def render_view(
    points: jnp.ndarray,
    intensities: jnp.ndarray,
    stds: jnp.ndarray,
    mouse_world_pos: jnp.ndarray,
    view_center: jnp.ndarray,
    view_size: jnp.ndarray,
    width: int,
    height: int
) -> jnp.ndarray:
    """Render gaussians for a given view and mouse position (all in world coordinates)."""
    # Add mouse gaussian
    all_points = jnp.concatenate([points, mouse_world_pos[None, :]])
    all_intensities = jnp.concatenate([intensities, jnp.array([1.0])])
    all_stds = jnp.concatenate([stds, jnp.array([20.0])])
    
    # Create coordinate grid (matching gaussian_renderer.py)
    x = jnp.linspace(-view_size[0]/2, view_size[0]/2, width)
    y = jnp.linspace(-view_size[1]/2, view_size[1]/2, height)
    X, Y = jnp.meshgrid(x, y, indexing='ij')  # Changed to 'ij' to match
    
    # Pre-compute transformed coordinates
    coords = jnp.stack([
        X + view_center[0],
        Y + view_center[1]
    ], axis=-1)
    
    # Vectorized gaussian computation
    def compute_gaussians(points, stds, intensities):
        diff = coords[None, :, :, :] - points[:, None, None, :]
        sq_distances = jnp.sum(diff**2, axis=-1)
        gaussians = intensities[:, None, None] * jnp.exp(
            -0.5 * sq_distances / (stds[:, None, None]**2)
        )
        return gaussians
    
    # Compute and sum all gaussians
    gaussians = compute_gaussians(all_points, all_stds, all_intensities)
    image = jnp.sum(gaussians, axis=0)
    
    return jnp.clip(image/2.0, 0.0, 1.0)

if __name__ == "__main__":
    # Create initial state
    points, intensities, stds = create_random_points()
    
    # Create renderer function that matches the expected interface
    def renderer(mouse_world_pos, view_center, view_size, width, height):
        # Convert numpy arrays to jax arrays
        mouse_world_pos = jnp.array(mouse_world_pos)
        view_center = jnp.array(view_center)
        view_size = jnp.array(view_size)
        
        image = render_view(points, intensities, stds, mouse_world_pos, 
                          view_center, view_size, width, height)
        return apply_colormap(image)
    
    # Run with pygame
    config = WindowConfig(
        width=1024,
        height=768,
        title="Simple Gaussian Renderer"
    )
    run_renderer(config, renderer)
