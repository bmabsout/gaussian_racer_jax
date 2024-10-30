import pygame
from src.simple_gaussian_renderer import render_gaussians, create_random_points
from src.pygame_utils import WindowConfig, run_app, PygameWindow
from src.gaussian_utils import apply_colormap
import jax.numpy as jnp

def render_frame(window: PygameWindow, state):
    """Render a single frame using the gaussian renderer."""
    points, intensities, stds = state
    width, height = window.size
    
    # Get view parameters
    center = jnp.array(window.viewport.position)
    size = jnp.array([width, height]) / window.viewport.zoom
    
    # Add mouse cursor
    x, y = pygame.mouse.get_pos()
    screen_to_world = lambda p: (p - jnp.array([width/2, height/2])) / window.viewport.zoom + center
    mouse_world = screen_to_world(jnp.array([x, y]))
    
    all_points = jnp.concatenate([points, mouse_world[None, :]])
    all_intensities = jnp.concatenate([intensities, jnp.array([1.0])])
    all_stds = jnp.concatenate([stds, jnp.array([20.0])])
    
    # Render
    image = render_gaussians(
        all_points, all_intensities, all_stds,
        center, size, width, height
    )
    colored = apply_colormap(image)
    
    # Display
    window.display_image(colored)
    window.display_fps()
    window.render_text("Click and drag to move, Scroll to zoom", (10, 50))

def run_visualization(width: int = 1024, height: int = 768):
    """Run the visualization."""
    config = WindowConfig(width=width, height=height, title="Simple Gaussian Renderer")
    state = create_random_points(n_points=2000, spread=500.0)
    run_app(config, lambda window: render_frame(window, state))

if __name__ == "__main__":
    run_visualization() 