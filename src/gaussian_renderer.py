import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
from functools import partial
from src.pygame_utils import WindowConfig, run_renderer
from src.gaussian_utils import apply_colormap

@jax.jit
def single_blur_pass(image: jnp.ndarray) -> jnp.ndarray:
    """Apply a single blur pass."""
    kernel = jnp.array([0.25, 0.5, 0.25])
    
    # Horizontal blur
    padded = jnp.pad(image, ((0, 0), (1, 1)), mode='edge')
    temp = (
        padded[:, :-2] * kernel[0] +
        padded[:, 1:-1] * kernel[1] +
        padded[:, 2:] * kernel[2]
    )
    
    # Vertical blur
    padded = jnp.pad(temp, ((1, 1), (0, 0)), mode='edge')
    result = (
        padded[:-2, :] * kernel[0] +
        padded[1:-1, :] * kernel[1] +
        padded[2:, :] * kernel[2]
    )
    
    return result

@jax.jit
def fast_blur(image: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    """Fast approximate gaussian blur using simple box blur."""
    def blur_body(i, acc):
        return single_blur_pass(acc)
    
    # Convert sigma to number of passes (1-4)
    n_passes = jnp.clip(jnp.floor(sigma), 1, 4).astype(jnp.int32)
    
    # Apply multiple blur passes
    return lax.fori_loop(0, n_passes, blur_body, image)

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
    all_stds = jnp.concatenate([stds * view_size[0]/width, jnp.array([3.0 * view_size[0]/width])])
    
    # Round points to nearest integer coordinates
    screen_points = (all_points - view_center) * width / view_size[0] + jnp.array([width/2, height/2])
    indices = jnp.round(screen_points).astype(jnp.int32)
    
    # Create empty image and accumulate intensities
    image = jnp.zeros((width, height))
    
    # Clamp indices to valid range
    x_indices = jnp.clip(indices[:, 0], 0, width-1).astype(jnp.int32)
    y_indices = jnp.clip(indices[:, 1], 0, height-1).astype(jnp.int32)
    
    # Scale intensities based on std
    scaled_intensities = all_intensities / jnp.sqrt(all_stds + 1.0)
    
    # Accumulate intensities at integer coordinates
    image = image.at[x_indices, y_indices].add(scaled_intensities)
    
    # Apply blur and normalize
    image = fast_blur(image, 2.0)
    return jnp.clip(image * 2.0, 0.0, 1.0)

def create_random_points(n_points: int = 50000, spread: float = 500.0):
    """Create random gaussian points in world space."""
    key = jax.random.PRNGKey(0)
    k1, k2 = random.split(key)
    
    points = random.normal(k1, shape=(n_points, 2)) * spread
    intensities = jnp.ones(n_points) * 0.5
    stds = jnp.exp(random.normal(k2, shape=(n_points,))) * 1.0
    
    return points, intensities, stds

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
        title="Fast Gaussian Renderer"
    )
    run_renderer(config, renderer)
