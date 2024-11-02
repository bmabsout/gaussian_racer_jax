from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Tuple

class Rectangle(NamedTuple):
    """A rectangle in world space."""
    center: jnp.ndarray
    width: float
    height: float

@dataclass(frozen=True)
class ViewTransform:
    """Handles coordinate transformations and view manipulation."""
    screen_size: jnp.ndarray  # (width, height)
    world_rect: Rectangle
    positions: jnp.ndarray    # Cache of world-space positions grid
    dragging: bool = False
    last_drag_pos: Optional[jnp.ndarray] = None
    
    @staticmethod
    def create(width: int, height: int, initial_scale: float = 2.0) -> 'ViewTransform':
        """Create initial view transform with given screen size."""
        screen_size = jnp.array([width, height])
        rect = Rectangle(
            center=jnp.zeros(2),
            width=width * initial_scale,
            height=height * initial_scale
        )
        transform = ViewTransform(
            screen_size=screen_size,
            world_rect=rect,
            positions=None,
            dragging=False,
            last_drag_pos=None
        )
        positions = transform.create_position_grid()
        return replace(transform, positions=positions)
    
    def screen_to_world(self, screen_pos: jnp.ndarray) -> jnp.ndarray:
        """Convert screen coordinates to world coordinates."""
        screen_scale = jnp.array([self.world_rect.width, self.world_rect.height]) / self.screen_size
        return (screen_pos - self.screen_size/2) * screen_scale + self.world_rect.center
    
    def world_to_screen(self, world_pos: jnp.ndarray) -> jnp.ndarray:
        """Convert world coordinates to screen coordinates."""
        screen_scale = self.screen_size / jnp.array([self.world_rect.width, self.world_rect.height])
        return (world_pos - self.world_rect.center) * screen_scale + self.screen_size/2
    
    def update(self, mouse_pos: jnp.ndarray) -> Tuple['ViewTransform', bool]:
        """Update view based on current mouse position."""
        if not self.dragging:
            return self, False
            
        if self.last_drag_pos is None:
            return replace(self, last_drag_pos=mouse_pos), False
            
        drag_delta = mouse_pos - self.last_drag_pos
        if jnp.any(drag_delta != 0):
            new_transform = self.move_by_screen_delta(drag_delta)
            return replace(new_transform, last_drag_pos=mouse_pos), True
            
        return replace(self, last_drag_pos=mouse_pos), False
    
    def handle_mouse_down(self, mouse_pos: jnp.ndarray) -> 'ViewTransform':
        """Start dragging from given screen position."""
        return replace(self, dragging=True, last_drag_pos=mouse_pos)
    
    def handle_mouse_up(self) -> 'ViewTransform':
        """Stop dragging."""
        return replace(self, dragging=False, last_drag_pos=None)
    
    def handle_scroll(self, mouse_pos: jnp.ndarray, scroll_y: float) -> 'ViewTransform':
        """Handle scroll wheel input."""
        zoom_factor = 1.1 if scroll_y > 0 else 1/1.1
        return self.zoom(zoom_factor, mouse_pos)
    
    def handle_resize(self, width: int, height: int) -> 'ViewTransform':
        """Handle window resize."""
        new_screen_size = jnp.array([width, height])
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
    
    def create_position_grid(self) -> jnp.ndarray:
        """Create a grid of world-space positions for rendering."""
        rect_size = jnp.array([self.world_rect.width, self.world_rect.height])
        width, height = int(self.screen_size[0]), int(self.screen_size[1])
        
        x = jnp.linspace(-rect_size[0]/2, rect_size[0]/2, width)
        y = jnp.linspace(-rect_size[1]/2, rect_size[1]/2, height)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        return jnp.stack([
            X + self.world_rect.center[0],
            Y + self.world_rect.center[1]
        ], axis=-1)
    
    def move_by_screen_delta(self, screen_delta: jnp.ndarray) -> 'ViewTransform':
        """Move view by a screen-space delta."""
        rect_size = jnp.array([self.world_rect.width, self.world_rect.height])
        screen_scale = rect_size / self.screen_size
        world_delta = screen_delta * screen_scale
        
        new_rect = Rectangle(
            center=self.world_rect.center - world_delta,
            width=self.world_rect.width,
            height=self.world_rect.height
        )
        
        if jnp.any(jnp.abs(world_delta) > 0.1 * rect_size.min()):
            new_positions = self.create_position_grid()
            return replace(self, world_rect=new_rect, positions=new_positions)
        return replace(self, world_rect=new_rect)
    
    def zoom(self, factor: float, pivot_screen_pos: jnp.ndarray) -> 'ViewTransform':
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
        
        if abs(1 - factor) > 0.05:
            final_transform = replace(self, world_rect=final_rect)
            new_positions = final_transform.create_position_grid()
            return replace(final_transform, positions=new_positions)
        return replace(self, world_rect=final_rect)