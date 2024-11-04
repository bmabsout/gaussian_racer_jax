from dataclasses import dataclass, replace
import numpy as np
from typing import NamedTuple, Optional, Tuple
from wgpu.gui.auto import WgpuCanvas

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
    canvas: WgpuCanvas
    dragging: bool = False
    last_drag_pos: Optional[np.ndarray] = None
    
    @staticmethod
    def create(width: int, height: int, canvas: WgpuCanvas, initial_scale: float = 2.0) -> 'ViewTransform':
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
            canvas=canvas,
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
    
    def handle_event(self, scroll_offset: tuple[float, float], mouse_pos: tuple[float, float], mouse_pressed: bool) -> Optional['ViewTransform']:
        """Handle input events."""
        # Handle mouse dragging
        if mouse_pressed and not self.dragging:
            # Start dragging
            return replace(self, dragging=True, last_drag_pos=np.array(mouse_pos))
        elif not mouse_pressed and self.dragging:
            # Stop dragging
            return replace(self, dragging=False, last_drag_pos=None)
        elif self.dragging:
            # Continue dragging
            if self.last_drag_pos is not None:
                current_pos = np.array(mouse_pos)
                delta = current_pos - self.last_drag_pos
                new_transform = self.move_by_screen_delta(delta)
                return replace(new_transform, last_drag_pos=current_pos)
        
        # Handle scroll
        _, scroll_y = scroll_offset
        if scroll_y != 0:
            zoom_factor = 0.9 if scroll_y > 0 else 1.1
            return self.zoom(zoom_factor)
        
        return None
    
    def handle_resize(self, width: int, height: int) -> 'ViewTransform':
        """Handle window resize event."""
        new_screen_size = np.array([width, height])
        
        # Keep the same world-space size but update aspect ratio
        aspect_ratio = width / height
        old_aspect_ratio = self.screen_size[0] / self.screen_size[1]
        
        if aspect_ratio > old_aspect_ratio:
            # Window got wider - adjust width
            new_width = self.world_rect.height * aspect_ratio
            new_rect = Rectangle(
                center=self.world_rect.center,
                width=new_width,
                height=self.world_rect.height
            )
        else:
            # Window got taller - adjust height
            new_height = self.world_rect.width / aspect_ratio
            new_rect = Rectangle(
                center=self.world_rect.center,
                width=self.world_rect.width,
                height=new_height
            )
        
        return replace(self,
            screen_size=new_screen_size,
            world_rect=new_rect
        )
    
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