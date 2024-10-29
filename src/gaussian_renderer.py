from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pygame
from functools import partial
import matplotlib.cm as cm

class GaussianParams(NamedTuple):
    """Parameters for a collection of 2D Gaussians."""
    positions: jnp.ndarray  # shape: (N, 2) for N Gaussians
    amplitudes: jnp.ndarray # shape: (N,)

def create_gaussian_patch(patch_size: int = 21, std: float = 3.0) -> jnp.ndarray:
    """Create a single Gaussian patch that we'll reuse."""
    center = patch_size // 2
    y, x = jnp.meshgrid(
        jnp.arange(patch_size),
        jnp.arange(patch_size),
        indexing='ij'
    )
    coords = jnp.stack([y - center, x - center], axis=-1)
    r_squared = jnp.sum(coords**2, axis=-1)
    return jnp.exp(-0.5 * r_squared / (std**2))

def apply_colormap(image: np.ndarray) -> np.ndarray:
    """Apply the inferno colormap to an image."""
    colormap = cm.get_cmap('inferno')
    colored = colormap(image)
    return (colored[:, :, :3] * 255).astype(np.uint8)

@partial(jax.jit, static_argnums=(3, 4))
def render_gaussians(
    positions: jnp.ndarray,
    amplitudes: jnp.ndarray,
    patch: jnp.ndarray,
    width: int,
    height: int
) -> jnp.ndarray:
    """Render all Gaussians by translating a pre-computed patch."""
    # Create coordinate grid for the entire image
    y, x = jnp.meshgrid(
        jnp.arange(height),
        jnp.arange(width),
        indexing='ij'
    )
    coords = jnp.stack([y, x], axis=-1)
    
    # Reshape for broadcasting
    coords = coords[None, :, :, :]
    positions = positions[:, None, None, :]
    amplitudes = amplitudes[:, None, None]
    
    # Compute distances
    diff = coords - positions
    distances = jnp.sum(diff**2, axis=-1)
    
    # Apply pre-computed patch values based on distances
    patch_radius = patch.shape[0] // 2
    patch_scale = (patch_radius ** 2) / 2
    values = amplitudes * jnp.exp(-distances / patch_scale)
    
    # Sum all gaussians
    image = jnp.sum(values, axis=0)
    
    # Normalize with a slight gamma correction for better visual contrast
    image = jnp.clip(image, 0.0, 5.0)/5.0
    return jnp.power(image, 0.8)  # Gamma correction

def run_visualization(width: int = 1024, height: int = 1024):
    """Run interactive visualization."""
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    
    # Create pre-computed Gaussian patch
    patch = create_gaussian_patch(patch_size=21, std=3.0)
    
    # Create random background gaussians
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    positions = random.uniform(
        k1,
        shape=(10000, 2),
        minval=jnp.array([0, 0]),
        maxval=jnp.array([height, width])
    )
    amplitudes = random.uniform(k2, shape=(10000,), minval=0.1, maxval=0.3)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Add mouse gaussian
        x, y = pygame.mouse.get_pos()
        all_positions = jnp.concatenate([positions, jnp.array([[y, x]])])
        all_amplitudes = jnp.concatenate([amplitudes, jnp.array([1.0])])
        
        # Render
        image = render_gaussians(all_positions, all_amplitudes, patch, width, height)
        
        # Convert to numpy and apply colormap
        image_np = np.array(image)
        colored_image = apply_colormap(image_np)
        
        # Create pygame surface
        surface = pygame.surfarray.make_surface(colored_image.transpose(1, 0, 2))
        screen.blit(surface, (0, 0))
        
        # Show FPS
        font = pygame.font.Font(None, 36)
        fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    run_visualization()
