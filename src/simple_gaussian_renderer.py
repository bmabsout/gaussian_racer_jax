from dataclasses import dataclass
from typing import Tuple, Optional, NamedTuple, List
import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
import pygame

@dataclass(frozen=True)
class ImageGrid:
    """Represents a 2D coordinate grid for image rendering."""
    height: int
    width: int
    coords: jnp.ndarray

    @staticmethod
    def create(image_size: Tuple[int, int] = (256, 256)) -> 'ImageGrid':
        """Factory method to create an ImageGrid."""
        height, width = image_size
        y, x = jnp.meshgrid(
            jnp.linspace(0, height-1, height),
            jnp.linspace(0, width-1, width),
            indexing='ij'
        )
        coords = jnp.stack([y, x], axis=-1)
        return ImageGrid(height=height, width=width, coords=coords)

class GaussianParams(NamedTuple):
    """Parameters defining a set of 2D Gaussians."""
    means: jnp.ndarray      # shape: (N, 2) for N Gaussians
    stds: jnp.ndarray       # shape: (N,) or (N, 1)
    amplitudes: jnp.ndarray # shape: (N,) or (N, 1)

    @staticmethod
    def create(
        means: jnp.ndarray,
        stds: jnp.ndarray,
        amplitudes: Optional[jnp.ndarray] = None
    ) -> 'GaussianParams':
        """Factory method to create GaussianParams with optional amplitudes."""
        if amplitudes is None:
            amplitudes = jnp.ones_like(stds)
        return GaussianParams(means, stds, amplitudes)

def compute_single_gaussian(
    coords: jnp.ndarray,
    mean: jnp.ndarray,
    std: float,
    amplitude: float = 1.0
) -> jnp.ndarray:
    """Compute a single 2D Gaussian."""
    diff = coords - mean
    exponent = -0.5 * jnp.sum(diff**2, axis=-1) / (std**2)
    return amplitude * jnp.exp(exponent)

@jax.jit
def _render_gaussians_impl(
    coords: jnp.ndarray,
    means: jnp.ndarray,
    stds: jnp.ndarray,
    amplitudes: jnp.ndarray
) -> jnp.ndarray:
    """Internal JIT-compiled implementation of gaussian rendering."""
    vectorized_gaussian = vmap(
        lambda m, s, a: compute_single_gaussian(coords, m, s, a)
    )
    gaussians = vectorized_gaussian(means, stds, amplitudes)
    image = jnp.sum(gaussians, axis=0)
    return jnp.clip(image, 0.0, 5.0) / 5.0

def render_gaussians(
    grid: ImageGrid,
    params: GaussianParams
) -> jnp.ndarray:
    """Render multiple 2D Gaussians onto a grid."""
    return _render_gaussians_impl(
        grid.coords,
        params.means,
        params.stds,
        params.amplitudes
    )

@dataclass(frozen=True)
class DisplayConfig:
    """Configuration for the display window"""
    width: int
    height: int
    std: float = 20.0
    amplitude: float = 1.0
    fps_cap: int = 60
    font_size: int = 36

@dataclass(frozen=True)
class DisplayState:
    """Current state of the display"""
    screen: pygame.Surface
    clock: pygame.time.Clock
    font: pygame.font.Font
    config: DisplayConfig
    grid: ImageGrid

    @staticmethod
    def create(config: DisplayConfig) -> 'DisplayState':
        """Initialize pygame and create initial display state"""
        pygame.init()
        screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption("Interactive Gaussian Renderer")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, config.font_size)
        grid = ImageGrid.create((config.height, config.width))
        return DisplayState(screen, clock, font, config, grid)

@dataclass(frozen=True)
class RenderState:
    """Current state of the renderer"""
    std: float
    amplitude: float
    mouse_pos: Tuple[int, int]
    background_gaussians: GaussianParams  # Add background gaussians

    @staticmethod
    def create(
        config: DisplayConfig,
        key: jax.random.PRNGKey,
        n_gaussians: int = 10000
    ) -> 'RenderState':
        """Create initial render state with random background gaussians"""
        # Split PRNG key for different random operations
        k1, k2, k3 = random.split(key, 3)
        
        # Generate random positions
        means = random.uniform(
            k1,
            shape=(n_gaussians, 2),
            minval=0,
            maxval=jnp.array([config.width, config.height])
        )
        
        # Small standard deviations
        stds = random.uniform(
            k2,
            shape=(n_gaussians,),
            minval=1.0,
            maxval=3.0
        )
        
        # Random amplitudes
        amplitudes = random.uniform(
            k3,
            shape=(n_gaussians,),
            minval=0.1,
            maxval=0.3
        )
        
        return RenderState(
            std=config.std,
            amplitude=config.amplitude,
            mouse_pos=(0, 0),
            background_gaussians=GaussianParams(means, stds, amplitudes)
        )

    def update(self, event: Optional[pygame.event.Event] = None) -> 'RenderState':
        """Return new state based on event"""
        new_std = self.std
        new_amplitude = self.amplitude
        
        if event and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                new_std = min(50, self.std + 1)
            elif event.key == pygame.K_DOWN:
                new_std = max(1, self.std - 1)
            elif event.key == pygame.K_RIGHT:
                new_amplitude = min(2.0, self.amplitude + 0.1)
            elif event.key == pygame.K_LEFT:
                new_amplitude = max(0.1, self.amplitude - 0.1)
        
        return RenderState(
            std=new_std,
            amplitude=new_amplitude,
            mouse_pos=pygame.mouse.get_pos(),
            background_gaussians=self.background_gaussians
        )

def update_display(
    display_state: DisplayState,
    image: jnp.ndarray
) -> None:
    """Update the display with the new image"""
    # Convert JAX array to pygame surface
    pygame_image = (image * 255).astype(np.uint8)
    colormap = pygame.surfarray.make_surface(
        np.stack([pygame_image] * 3, axis=-1)
    )
    
    # Display the image
    display_state.screen.blit(colormap, (0, 0))
    
    # Display FPS
    fps = display_state.clock.get_fps()
    fps_text = display_state.font.render(
        f"FPS: {fps:.1f}", True, (255, 255, 255)
    )
    display_state.screen.blit(fps_text, (10, 10))
    
    pygame.display.flip()

def run_renderer(config: DisplayConfig):
    """Main rendering loop"""
    display_state = DisplayState.create(config)
    
    # Initialize with random seed
    key = random.PRNGKey(0)
    render_state = RenderState.create(config, key)
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            render_state = render_state.update(event)
        
        # Update state based on current mouse position
        render_state = render_state.update()
        
        # Create gaussian parameters for mouse cursor
        x, y = render_state.mouse_pos
        mouse_params = GaussianParams.create(
            means=jnp.array([[x, y]]),
            stds=jnp.array([render_state.std]),
            amplitudes=jnp.array([render_state.amplitude])
        )
        
        # Combine background and mouse gaussians
        combined_params = GaussianParams(
            means=jnp.concatenate([
                render_state.background_gaussians.means,
                mouse_params.means
            ]),
            stds=jnp.concatenate([
                render_state.background_gaussians.stds,
                mouse_params.stds
            ]),
            amplitudes=jnp.concatenate([
                render_state.background_gaussians.amplitudes,
                mouse_params.amplitudes
            ])
        )
        
        # Render combined gaussians
        image = render_gaussians(display_state.grid, combined_params)
        
        # Update display
        update_display(display_state, image)
        
        # Cap framerate
        display_state.clock.tick(config.fps_cap)

    pygame.quit()

if __name__ == "__main__":
    config = DisplayConfig(
        width=1024,
        height=1024,  # Larger window
        std=5.0,     # Smaller std for mouse gaussian
        amplitude=1.0,
        fps_cap=60
    )
    run_renderer(config)
