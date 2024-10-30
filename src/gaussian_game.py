import numpy as np
from gaussian_utils import apply_colormap
import pygame
from dataclasses import dataclass
from src.pygame_utils import Scene
from src.simple_gaussian_renderer import Gaussians, render_view, create_random_gaussians

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
        self.max_added = 1000
        self.added = Gaussians(
            pos=np.zeros((self.max_added, 2)),
            std=np.zeros(self.max_added),
            intensity=np.zeros(self.max_added)
        )
        self.current_index = 0
        self.active_gaussians = np.zeros(self.max_added, dtype=bool)
        
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
            # Add new gaussian
            idx = self.current_index % self.max_added
            world_pos = self.view.screen_to_world(mouse_pos, self.screen_size)
            
            self.added = Gaussians(
                pos=self.added.pos.at[idx].set(world_pos),
                std=self.added.std.at[idx].set(20.0),
                intensity=self.added.intensity.at[idx].set(1.0)
            )
            self.active_gaussians = self.active_gaussians.at[idx].set(True)
            self.current_index += 1
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
        
        # Filter active gaussians
        active_pos = self.added.pos[self.active_gaussians]
        active_std = self.added.std[self.active_gaussians]
        active_intensity = self.added.intensity[self.active_gaussians]
        active_added = Gaussians(active_pos, active_std, active_intensity)
        
        # Compose all gaussians
        all_gaussians = Gaussians.compose(self.background, active_added, mouse_gaussian)
        
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