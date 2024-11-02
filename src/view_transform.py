from dataclasses import dataclass, replace
import numpy as np
from typing import NamedTuple, Optional, Tuple
import glfw

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
    
    def update(self, mouse_pos: np.ndarray) -> Tuple['ViewTransform', bool]:
        """Update view based on current mouse position."""
        if not self.dragging:
            return self, False
            
        if self.last_drag_pos is None:
            return replace(self, last_drag_pos=mouse_pos), False
            
        drag_delta = mouse_pos - self.last_drag_pos
        if np.any(drag_delta != 0):
            new_transform = self.move_by_screen_delta(drag_delta)
            return replace(new_transform, last_drag_pos=mouse_pos), True
            
        return replace(self, last_drag_pos=mouse_pos), False
    
    def handle_event(self, window: int, scroll_offset: tuple[float, float]) -> Optional['ViewTransform']:
        """Handle GLFW window events."""
        # Handle scroll first
        _, scroll_y = scroll_offset
        if scroll_y != 0:
            mouse_pos = np.array(glfw.get_cursor_pos(window))
            zoom_factor = 0.9 if scroll_y > 0 else 1.1
            return self.zoom(zoom_factor, mouse_pos)
        
        # Handle mouse buttons
        left_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        if left_pressed and not self.dragging:
            mouse_pos = np.array(glfw.get_cursor_pos(window))
            return self.handle_mouse_down(mouse_pos)
        elif not left_pressed and self.dragging:
            return self.handle_mouse_up()
        
        # Handle window resize
        width, height = glfw.get_window_size(window)
        current_size = np.array([width, height])
        if not np.array_equal(current_size, self.screen_size):
            return self.handle_resize(width, height)
        
        return None
    
    def handle_mouse_down(self, mouse_pos: np.ndarray) -> 'ViewTransform':
        """Start dragging from given screen position."""
        return replace(self, dragging=True, last_drag_pos=mouse_pos)
    
    def handle_mouse_up(self) -> 'ViewTransform':
        """Stop dragging."""
        return replace(self, dragging=False, last_drag_pos=None)
    
    def handle_resize(self, width: int, height: int) -> 'ViewTransform':
        """Handle window resize."""
        new_screen_size = np.array([width, height])
        aspect_ratio = width / height
        old_aspect_ratio = self.screen_size[0] / self.screen_size[1]
        
        if aspect_ratio > old_aspect_ratio:
            new_width = self.world_rect.height * aspect_ratio
            new_rect = Rectangle(
                center=self.world_rect.center,
                width=new_width,
                height=self.world_rect.height
            )
        else:
            new_height = self.world_rect.width / aspect_ratio
            new_rect = Rectangle(
                center=self.world_rect.center,
                width=self.world_rect.width,
                height=new_height
            )
        
        new_transform = replace(self,
            screen_size=new_screen_size,
            world_rect=new_rect
        )
        return replace(new_transform, positions=new_transform.create_position_grid())
    
    def move_by_screen_delta(self, screen_delta: np.ndarray) -> 'ViewTransform':
        """Move view by a screen-space delta."""
        rect_size = np.array([self.world_rect.width, self.world_rect.height])
        screen_scale = rect_size / self.screen_size
        world_delta = screen_delta * screen_scale * np.array([1.0, -1.0])
        
        new_rect = Rectangle(
            center=self.world_rect.center - world_delta,
            width=self.world_rect.width,
            height=self.world_rect.height
        )
        
        return replace(self, world_rect=new_rect)
    
    def zoom(self, factor: float, pivot_screen_pos: np.ndarray) -> 'ViewTransform':
        """Zoom view, keeping the pivot point (in screen space) fixed in world space."""
        old_world_pos = self.screen_to_world(pivot_screen_pos)
        
        new_rect = Rectangle(
            center=self.world_rect.center,
            width=self.world_rect.width * factor,
            height=self.world_rect.height * factor
        )
        
        new_transform = replace(self, world_rect=new_rect)
        new_world_pos = new_transform.screen_to_world(pivot_screen_pos)
        
        delta = new_world_pos - old_world_pos
        final_rect = Rectangle(
            center=new_rect.center - delta,
            width=new_rect.width,
            height=new_rect.height
        )
        
        return replace(self, world_rect=final_rect)