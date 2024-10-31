import numpy as np
import pygame
from dataclasses import dataclass, replace
from typing import Optional, Tuple
from src.pygame_utils import WindowConfig, SceneState
from src.simple_gaussian_renderer import Gaussians, render_view, create_random_gaussians, apply_colormap
import jax.numpy as jnp

@dataclass(frozen=True)
class Rectangle:
    """A rectangle in world space that we want to view."""
    center: np.ndarray
    width: float
    height: float
    
    def move_by(self, delta: np.ndarray) -> 'Rectangle':
        """Move rectangle by delta in world coordinates."""
        return replace(self, center=self.center + delta)
    
    def scale_by(self, factor: float) -> 'Rectangle':
        """Scale rectangle by factor."""
        return Rectangle(
            center=self.center,
            width=self.width * factor,
            height=self.height * factor
        )

@dataclass(frozen=True)
class MouseState:
    """Mouse state with screen and world coordinates."""
    screen_pos: np.ndarray
    dragging: bool = False
    last_drag_pos: Optional[np.ndarray] = None

    def handle_event(self, event: pygame.event.Event) -> Optional['MouseState']:
        """Return new state if event is handled, None otherwise."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.start_drag()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            return self.stop_drag()
        return None

    def start_drag(self) -> 'MouseState':
        return replace(self, dragging=True, last_drag_pos=self.screen_pos)
    
    def stop_drag(self) -> 'MouseState':
        return replace(self, dragging=False, last_drag_pos=None)
    
    def update(self, new_screen_pos: np.ndarray) -> Tuple['MouseState', Optional[np.ndarray]]:
        """Update state and return (new_state, drag_delta)."""
        if self.dragging:
            new_state = replace(self, screen_pos=new_screen_pos, last_drag_pos=self.screen_pos)
            drag_delta = new_screen_pos - self.screen_pos
            return new_state, drag_delta
        return replace(self, screen_pos=new_screen_pos), None

@dataclass(frozen=True)
class ViewState:
    """Camera view state."""
    rect: Rectangle      # The world space we want to see
    resolution: np.ndarray  # Current render resolution (width, height)
    
    @staticmethod
    def create(width: int, height: int) -> 'ViewState':
        resolution = np.array([width, height])
        return ViewState(
            rect=Rectangle(
                center=np.zeros(2),
                width=width * 2.0,  # Initial view is 2x screen size
                height=height * 2.0
            ),
            resolution=resolution
        )
    
    def handle_event(self, event: pygame.event.Event) -> Optional['ViewState']:
        """Return new state if event is handled, None otherwise."""
        if event.type == pygame.VIDEORESIZE:
            return replace(self, resolution=np.array([event.w, event.h]))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Mouse wheel up
                return replace(self, rect=self.rect.scale_by(1/1.1))
            elif event.button == 5:  # Mouse wheel down
                return replace(self, rect=self.rect.scale_by(1.1))
        return None

    def screen_to_world(self, screen_pos: np.ndarray) -> np.ndarray:
        """Convert screen coordinates to world coordinates."""
        screen_scale = np.array([self.rect.width, self.rect.height]) / self.resolution
        return (screen_pos - self.resolution/2) * screen_scale + self.rect.center
    
    def move_by(self, screen_delta: np.ndarray) -> 'ViewState':
        """Move view by screen delta."""
        world_delta = screen_delta * np.array([self.rect.width, self.rect.height]) / self.resolution
        return replace(self, rect=self.rect.move_by(-world_delta))

@dataclass(frozen=True)
class GaussianState:
    """State of all gaussians."""
    background: Gaussians
    added: Gaussians
    last_add_time: float = 0.0
    
    @staticmethod
    def create() -> 'GaussianState':
        return GaussianState(
            background=create_random_gaussians(),
            added=Gaussians(
                pos=jnp.zeros((0, 2)),
                std=jnp.zeros(0),
                intensity=jnp.zeros(0)
            )
        )
    
    def update(self, world_pos: np.ndarray) -> Optional['GaussianState']:
        """Update state, possibly adding new gaussian."""
        current_time = pygame.time.get_ticks() / 1000.0
        if (pygame.key.get_mods() & pygame.KMOD_SHIFT and 
            current_time - self.last_add_time >= 1/60):
            return self.add_gaussian(world_pos, current_time)
        return None
    
    def add_gaussian(self, pos: np.ndarray, current_time: float) -> 'GaussianState':
        """Add a new gaussian."""
        new_gaussian = Gaussians(
            pos=jnp.array([pos]),
            std=jnp.array([20.0]),
            intensity=jnp.array([1.0])
        )
        return replace(self,
            added=Gaussians.compose(self.added, new_gaussian),
            last_add_time=current_time
        )
    
    def get_all_gaussians(self, mouse_world_pos: np.ndarray) -> Gaussians:
        """Get all gaussians including mouse cursor."""
        mouse_gaussian = Gaussians(
            pos=np.array([mouse_world_pos]),
            std=np.array([20.0]),
            intensity=np.array([1.0])
        )
        return Gaussians.compose(self.background, self.added, mouse_gaussian)

@dataclass(frozen=True)
class GameState(SceneState):
    """Complete game state."""
    view: ViewState
    mouse: MouseState
    gaussians: GaussianState
    
    @staticmethod
    def create(width: int, height: int) -> 'GameState':
        return GameState(
            view=ViewState.create(width, height),
            mouse=MouseState(screen_pos=np.zeros(2)),
            gaussians=GaussianState.create()
        )
    
    def update(self, dt: float) -> 'GameState':
        # Update mouse and get drag delta if any
        new_mouse, drag_delta = self.mouse.update(np.array(pygame.mouse.get_pos()))
        
        # Update view if dragging
        new_view = self.view
        if drag_delta is not None:
            new_view = self.view.move_by(drag_delta)
        
        # Update gaussians
        world_pos = self.view.screen_to_world(new_mouse.screen_pos)
        new_gaussians = self.gaussians.update(world_pos)
        
        return replace(self,
            view=new_view,
            mouse=new_mouse,
            gaussians=new_gaussians if new_gaussians is not None else self.gaussians
        )
    
    def handle_event(self, event: pygame.event.Event) -> 'GameState':
        # Let each component handle its events
        new_mouse = self.mouse.handle_event(event)
        new_view = self.view.handle_event(event)
        
        return replace(self,
            mouse=new_mouse if new_mouse is not None else self.mouse,
            view=new_view if new_view is not None else self.view,
        )
    
    def render(self) -> np.ndarray:
        # Get current mouse position in world space
        mouse_world_pos = self.view.screen_to_world(self.mouse.screen_pos)
        
        # Get all gaussians including mouse cursor
        all_gaussians = self.gaussians.get_all_gaussians(mouse_world_pos)
        
        # Render view
        width, height = self.view.resolution
        image = render_view(
            all_gaussians,
            self.view.rect.center,
            np.array([self.view.rect.width, self.view.rect.height]),
            width,
            height
        )
        
        return apply_colormap(np.array(image))

if __name__ == "__main__":
    from src.pygame_utils import GameEngine, WindowConfig
    
    config = WindowConfig(
        width=1024,
        height=768,
        title="Gaussian Game"
    )
    
    engine = GameEngine.create(config)
    game_state = GameState.create(config.width, config.height)
    engine.run(game_state)