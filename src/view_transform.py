from dataclasses import dataclass, replace
import numpy as np
from typing import NamedTuple, Optional

class Rectangle(NamedTuple):
    center: np.ndarray
    width: float
    height: float
    
    def resize(self, scale_x: float, scale_y: float) -> 'Rectangle':
        return Rectangle(
            center=self.center,
            width=self.width * scale_x,
            height=self.height * scale_y
        )

def transform(point: np.ndarray, from_rect: Rectangle, to_rect: Rectangle) -> np.ndarray:
    """Transform a point from one rectangle's space to another's."""
    # First transform to normalized space [-1, 1]
    normalized = (point - from_rect.center) / np.array([from_rect.width/2, from_rect.height/2])
    # Then transform to target space
    return normalized * np.array([to_rect.width/2, to_rect.height/2]) + to_rect.center

@dataclass(frozen=True)
class ViewTransform:
    screen_rect: Rectangle  # Screen space rectangle
    world_rect: Rectangle   # World space rectangle
    dragging: bool = False
    last_drag_pos: Optional[np.ndarray] = None
    
    @staticmethod
    def create(width: int, height: int, initial_scale: float = 2.0) -> 'ViewTransform':
        screen_rect = Rectangle(
            center=np.array([width/2, height/2]),
            width=width,
            height=height
        )
        world_rect = Rectangle(
            center=np.zeros(2),
            width=width * initial_scale,
            height=height * initial_scale
        )
        return ViewTransform(screen_rect=screen_rect, world_rect=world_rect)
    
    def screen_to_world(self, screen_pos: np.ndarray) -> np.ndarray:
        # Flip y coordinate (screen y is down, world y is up)
        flipped_pos = np.array([screen_pos[0], self.screen_rect.height - screen_pos[1]])
        return transform(flipped_pos, self.screen_rect, self.world_rect)
    
    def world_to_screen(self, world_pos: np.ndarray) -> np.ndarray:
        screen_pos = transform(world_pos, self.world_rect, self.screen_rect)
        # Flip y coordinate back
        return np.array([screen_pos[0], self.screen_rect.height - screen_pos[1]])
    
    def handle_event(self, scroll_offset: tuple[float, float], mouse_pos: tuple[float, float], mouse_pressed: bool) -> Optional['ViewTransform']:
        if mouse_pressed and not self.dragging:
            return replace(self, dragging=True, last_drag_pos=np.array(mouse_pos))
        
        if not mouse_pressed and self.dragging:
            return replace(self, dragging=False, last_drag_pos=None)
        
        if self.dragging and self.last_drag_pos is not None:
            current_pos = np.array(mouse_pos)
            delta = current_pos - self.last_drag_pos
            new_transform = self._move_by_screen_delta(delta)
            return replace(new_transform, last_drag_pos=current_pos)
        
        _, scroll_y = scroll_offset
        if scroll_y != 0:
            zoom = 0.9 if scroll_y > 0 else 1.1
            return self._zoom(zoom)
        
        return None
    
    def handle_resize(self, width: int, height: int) -> 'ViewTransform':
        old_size = np.array([self.screen_rect.width, self.screen_rect.height])
        scale_x = width / old_size[0]
        scale_y = height / old_size[1]
        
        new_screen_rect = Rectangle(
            center=np.array([width/2, height/2]),
            width=width,
            height=height
        )
        new_world_rect = self.world_rect.resize(scale_x, scale_y)
        
        return replace(self, screen_rect=new_screen_rect, world_rect=new_world_rect)
    
    def _zoom(self, factor: float) -> 'ViewTransform':
        new_world_rect = Rectangle(
            center=self.world_rect.center,
            width=self.world_rect.width * factor,
            height=self.world_rect.height * factor
        )
        return replace(self, world_rect=new_world_rect)
    
    def _move_by_screen_delta(self, screen_delta: np.ndarray) -> 'ViewTransform':
        scale = np.array([self.world_rect.width/self.screen_rect.width,
                         self.world_rect.height/self.screen_rect.height])
        world_delta = screen_delta * scale * np.array([1.0, -1.0])  # Flip y
        
        new_world_rect = Rectangle(
            center=self.world_rect.center - world_delta,
            width=self.world_rect.width,
            height=self.world_rect.height
        )
        return replace(self, world_rect=new_world_rect)