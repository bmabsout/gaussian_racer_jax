import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
import pygame
import matplotlib.cm as cm
from functools import partial

# Pre-compute colormap
COLORMAP = (cm.get_cmap('inferno')(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

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

def run_visualization(initial_width: int = 1024, initial_height: int = 768):
    """Run visualization with minimal overhead."""
    pygame.init()
    
    # Initialize resizable window
    screen = pygame.display.set_mode((initial_width, initial_height), pygame.RESIZABLE)
    pygame.display.set_caption("Gaussian Renderer")
    clock = pygame.time.Clock()
    
    # Generate random points in world space
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    n_points = 50000
    
    points = random.normal(k1, shape=(n_points, 2)) * 500
    intensities = jnp.ones(n_points) * 0.5
    stds = jnp.exp(random.normal(k2, shape=(n_points,))) * 1.0
    
    # Camera state
    camera_pos = jnp.zeros(2)
    zoom = 0.5
    dragging = False
    last_mouse_pos = None
    
    # Current window dimensions
    width, height = initial_width, initial_height
    surface = pygame.Surface((width, height))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.size
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                surface = pygame.Surface((width, height))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    dragging = True
                    last_mouse_pos = np.array(pygame.mouse.get_pos())
                elif event.button == 4:  # Mouse wheel up
                    zoom *= 1.1
                elif event.button == 5:  # Mouse wheel down
                    zoom /= 1.1
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
        
        # Handle camera movement
        if dragging and last_mouse_pos is not None:
            current_mouse_pos = np.array(pygame.mouse.get_pos())
            delta = (current_mouse_pos - last_mouse_pos) / zoom
            camera_pos = camera_pos + delta
            last_mouse_pos = current_mouse_pos
        
        # Transform points to screen space
        screen_center = jnp.array([width/2, height/2])
        visible_points = (points + camera_pos) * zoom + screen_center
        
        # Add mouse point (in screen space)
        x, y = pygame.mouse.get_pos()
        all_points = jnp.concatenate([visible_points, jnp.array([[x, y]])])
        all_intensities = jnp.concatenate([intensities, jnp.array([1.0])])
        all_stds = jnp.concatenate([stds * zoom, jnp.array([3.0 * zoom])])  # Fixed mouse std
        
        # Render
        image = splat_points(all_points, all_intensities, all_stds, width, height)
        image = fast_blur(image, 2.0)  # Fixed blur amount
        
        # Normalize and clip
        image = jnp.clip(image * 2.0, 0.0, 1.0)
        
        # Display
        colored = COLORMAP[(image * 255).astype(np.uint8)]
        surface = pygame.surfarray.make_surface(colored)
        screen.blit(surface, (0, 0))
        
        # Show FPS and controls
        font = pygame.font.Font(None, 36)
        fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, (255, 255, 255))
        controls_text = font.render("Click and drag to move, Scroll to zoom", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))
        screen.blit(controls_text, (10, 50))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    run_visualization()
