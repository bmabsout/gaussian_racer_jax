import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
import matplotlib.cm as cm
from functools import partial
from dataclasses import dataclass
from typing import Tuple, Optional

# Pre-compute colormap
COLORMAP = (cm.get_cmap('inferno')(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

@dataclass(frozen=True)
class Camera:
    """Camera state for 2D visualization."""
    position: jnp.ndarray  # [x, y]
    zoom: float = 1.0

    @staticmethod
    def create() -> 'Camera':
        return Camera(position=jnp.zeros(2), zoom=0.5)
    
    def move(self, delta: np.ndarray) -> 'Camera':
        """Move camera by delta (in world space)."""
        return Camera(
            position=self.position + delta,
            zoom=self.zoom
        )
    
    def adjust_zoom(self, factor: float) -> 'Camera':
        """Multiply zoom by factor."""
        return Camera(
            position=self.position,
            zoom=self.zoom * factor
        )

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

@partial(jax.jit, static_argnums=(3, 4))
def splat_points(
    points: jnp.ndarray,
    intensities: jnp.ndarray,
    stds: jnp.ndarray,
    width: int,
    height: int
) -> jnp.ndarray:
    """Splat points onto a grid using nearest-neighbor."""
    # Round points to nearest integer coordinates
    indices = jnp.round(points).astype(jnp.int32)
    
    # Create empty image and accumulate intensities
    image = jnp.zeros((width, height))
    
    # Clamp indices to valid range
    x_indices = jnp.clip(indices[:, 0], 0, width-1).astype(jnp.int32)
    y_indices = jnp.clip(indices[:, 1], 0, height-1).astype(jnp.int32)
    
    # Scale intensities based on std
    scaled_intensities = intensities / jnp.sqrt(stds + 1.0)
    
    # Accumulate intensities at integer coordinates
    return image.at[x_indices, y_indices].add(scaled_intensities)

def generate_random_points(
    n_points: int,
    spread: float = 500.0,
    key: Optional[random.PRNGKey] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate random points with intensities and standard deviations."""
    if key is None:
        key = random.PRNGKey(0)
    
    k1, k2 = random.split(key)
    
    points = random.normal(k1, shape=(n_points, 2)) * spread
    intensities = jnp.ones(n_points) * 0.5
    stds = jnp.exp(random.normal(k2, shape=(n_points,))) * 1.0
    
    return points, intensities, stds

def world_to_screen(
    points: jnp.ndarray,
    camera: Camera,
    screen_size: Tuple[int, int]
) -> jnp.ndarray:
    """Transform points from world space to screen space."""
    width, height = screen_size
    screen_center = jnp.array([width/2, height/2])
    return (points + camera.position) * camera.zoom + screen_center

def apply_colormap(image: jnp.ndarray) -> np.ndarray:
    """Apply the inferno colormap to an image."""
    return COLORMAP[(image * 255).astype(np.uint8)] 