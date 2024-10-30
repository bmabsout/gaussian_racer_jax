import numpy as np
import pygame
from dataclasses import dataclass
from src.pygame_utils import Scene
from src.simple_gaussian_renderer import Gaussians, render_view, create_random_gaussians, apply_colormap
import jax.numpy as jnp

@dataclass
class ViewState:
    """Camera/view state."""
    center: np.ndarray
    zoom: float = 0.5

    def screen_to_world(self, screen_pos: np.ndarray, screen_size: np.ndarray) -> np.ndarray:
        """Convert screen coordinates to world coordinates."""
        return (screen_pos - screen_size/2) / self.zoom + self.center

class GaussianGame(Scene):
    def __init__(self, width: int, height: int):
        # View state
        self.view = ViewState(center=np.zeros(2))
        self.screen_size = np.array([width, height])
        
        # Mouse state
        self.dragging = False
        self.last_mouse_pos = None
        
        # Gaussian state
        self.background = create_random_gaussians()
        self.added_gaussians = Gaussians(
            pos=jnp.zeros((0, 2)),
            std=jnp.zeros(0),
            intensity=jnp.zeros(0)
        )
        
        # Rate limiting for adding gaussians
        self.last_add_time = 0
        self.add_interval = 1/60
    
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.dragging = True
                self.last_mouse_pos = np.array(pygame.mouse.get_pos())
            elif event.button == 4:  # Mouse wheel up
                self.view.zoom *= 1.1
            elif event.button == 5:  # Mouse wheel down
                self.view.zoom /= 1.1
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
            self.last_mouse_pos = None
        elif event.type == pygame.VIDEORESIZE:
            self.screen_size = np.array([event.w, event.h])
    
    def update(self, dt: float) -> None:
        mouse_pos = np.array(pygame.mouse.get_pos())
        
        # Handle dragging
        if self.dragging and self.last_mouse_pos is not None:
            delta = mouse_pos - self.last_mouse_pos
            self.view.center -= delta / self.view.zoom
            self.last_mouse_pos = mouse_pos
        
        # Add gaussians when holding shift
        current_time = pygame.time.get_ticks() / 1000.0
        if (pygame.key.get_mods() & pygame.KMOD_SHIFT and 
            current_time - self.last_add_time >= self.add_interval):
            # Create new gaussian
            world_pos = self.view.screen_to_world(mouse_pos, self.screen_size)
            new_gaussian = Gaussians(
                pos=jnp.array([world_pos]),
                std=jnp.array([20.0]),
                intensity=jnp.array([1.0])
            )
            
            # Add to existing gaussians
            self.added_gaussians = Gaussians.compose(self.added_gaussians, new_gaussian)
            self.last_add_time = current_time
    
    def render(self) -> np.ndarray:
        # Get current mouse position in world space
        mouse_pos = self.view.screen_to_world(
            np.array(pygame.mouse.get_pos()),
            self.screen_size
        )
        
        # Create mouse gaussian
        mouse_gaussian = Gaussians(
            pos=np.array([mouse_pos]),
            std=np.array([20.0]),
            intensity=np.array([1.0])
        )
        
        # Compose all gaussians
        all_gaussians = Gaussians.compose(self.background, self.added_gaussians, mouse_gaussian)
        
        # Render view
        width, height = self.screen_size
        image = render_view(
            all_gaussians,
            self.view.center,
            self.screen_size / self.view.zoom,
            width,
            height
        )
        
        # Convert to numpy and apply colormap
        return apply_colormap(np.array(image))

if __name__ == "__main__":
    from src.pygame_utils import GameEngine, WindowConfig
    
    config = WindowConfig(
        width=1024,
        height=768,
        title="Gaussian Game"
    )
    
    engine = GameEngine(config)
    game = GaussianGame(config.width, config.height)
    engine.run(game) 