from dataclasses import dataclass, replace
import numpy as np
from typing import NamedTuple, Optional, Tuple

class Rectangle(NamedTuple):
    """A rectangle in world space."""
    center: np.ndarray
    width: float
    height: float

@dataclass(frozen=True)
class ViewTransform:
    """Handles coordinate transformations and view manipulation."""
    screen_size: np.ndarray  # (width, height)
    world_rect: Rectangle
    dragging: bool = False
    last_drag_pos: Optional[np.ndarray] = None
    
    @staticmethod
    def create(width: int, height: int, initial_scale: float = 2.0) -> 'ViewTransform':
        """Create initial view transform with given screen size."""
        screen_size = np.array([width, height])
        rect = Rectangle(
            center=np.zeros(2),
            width=width * initial_scale,
            height=height * initial_scale
        )
        return ViewTransform(
            screen_size=screen_size,
            world_rect=rect,
            dragging=False,
            last_drag_pos=None
        )
    
    def screen_to_world(self, screen_pos: np.ndarray) -> np.ndarray:
        """Convert screen coordinates to world coordinates."""
        screen_scale = np.array([self.world_rect.width, self.world_rect.height]) / self.screen_size
        return (screen_pos - self.screen_size/2) * screen_scale + self.world_rect.center
    
    def world_to_screen(self, world_pos: np.ndarray) -> np.ndarray:
        """Convert world coordinates to screen coordinates."""
        screen_scale = self.screen_size / np.array([self.world_rect.width, self.world_rect.height])
        return (world_pos - self.world_rect.center) * screen_scale + self.screen_size/2
    
    def handle_event(self, scroll_offset: tuple[float, float]) -> Optional['ViewTransform']:
        """Handle scroll events."""
        _, scroll_y = scroll_offset
        if scroll_y != 0:
            zoom_factor = 0.9 if scroll_y > 0 else 1.1
            return self.zoom(zoom_factor)
        return None
    
    def zoom(self, factor: float) -> 'ViewTransform':
        """Zoom view."""
        new_rect = Rectangle(
            center=self.world_rect.center,
            width=self.world_rect.width * factor,
            height=self.world_rect.height * factor
        )
        return replace(self, world_rect=new_rect)
    
    def move_by_screen_delta(self, screen_delta: np.ndarray) -> 'ViewTransform':
        """Move view by a screen-space delta."""
        rect_size = np.array([self.world_rect.width, self.world_rect.height])
        screen_scale = rect_size / self.screen_size
        world_delta = screen_delta * screen_scale * np.array([1.0, -1.0])  # Flip y-axis
        
        new_rect = Rectangle(
            center=self.world_rect.center - world_delta,
            width=self.world_rect.width,
            height=self.world_rect.height
        )
        
        return replace(self, world_rect=new_rect)