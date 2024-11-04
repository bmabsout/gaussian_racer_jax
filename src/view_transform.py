from dataclasses import dataclass, replace
import numpy as np
from typing import NamedTuple, Optional

class Rectangle(NamedTuple):
    center: np.ndarray  # (x, y)
    width: float
    height: float

@dataclass(frozen=True)
class ViewTransform:
    screen_size: np.ndarray  # (width, height)
    world_rect: Rectangle
    dragging: bool = False
    last_drag_pos: Optional[np.ndarray] = None
    
    @staticmethod
    def create(width: int, height: int, initial_scale: float = 2.0) -> 'ViewTransform':
        screen_size = np.array([width, height])
        rect = Rectangle(
            center=np.zeros(2),
            width=width * initial_scale,
            height=height * initial_scale
        )
        return ViewTransform(screen_size=screen_size, world_rect=rect)
    
    def screen_to_world(self, screen_pos: np.ndarray) -> np.ndarray:
        # Flip y coordinate (screen y is down, world y is up)
        flipped_pos = np.array([screen_pos[0], self.screen_size[1] - screen_pos[1]])
        scale = np.array([self.world_rect.width, self.world_rect.height]) / self.screen_size
        return (flipped_pos - self.screen_size/2) * scale + self.world_rect.center
    
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
        new_size = np.array([width, height])
        aspect = width / height
        old_aspect = self.screen_size[0] / self.screen_size[1]
        
        if aspect > old_aspect:
            new_rect = Rectangle(
                center=self.world_rect.center,
                width=self.world_rect.height * aspect,
                height=self.world_rect.height
            )
        else:
            new_rect = Rectangle(
                center=self.world_rect.center,
                width=self.world_rect.width,
                height=self.world_rect.width / aspect
            )
        
        return replace(self, screen_size=new_size, world_rect=new_rect)
    
    def _zoom(self, factor: float) -> 'ViewTransform':
        new_rect = Rectangle(
            center=self.world_rect.center,
            width=self.world_rect.width * factor,
            height=self.world_rect.height * factor
        )
        return replace(self, world_rect=new_rect)
    
    def _move_by_screen_delta(self, screen_delta: np.ndarray) -> 'ViewTransform':
        scale = np.array([self.world_rect.width, self.world_rect.height]) / self.screen_size
        world_delta = screen_delta * scale * np.array([1.0, -1.0])  # Flip y
        
        new_rect = Rectangle(
            center=self.world_rect.center - world_delta,
            width=self.world_rect.width,
            height=self.world_rect.height
        )
        return replace(self, world_rect=new_rect)