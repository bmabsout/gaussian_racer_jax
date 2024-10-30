import pygame
from dataclasses import dataclass
from typing import Tuple, Callable, Optional
import numpy as np

@dataclass
class WindowConfig:
    """Configuration for pygame window."""
    width: int
    height: int
    title: str = "Pygame Window"
    resizable: bool = True
    vsync: bool = True

@dataclass
class Viewport:
    """Represents a movable, zoomable viewport."""
    position: np.ndarray  # Center position
    zoom: float = 1.0
    
    @staticmethod
    def create() -> 'Viewport':
        return Viewport(position=np.zeros(2), zoom=0.5)
    
    def move(self, delta: np.ndarray) -> 'Viewport':
        return Viewport(position=self.position + delta, zoom=self.zoom)
    
    def adjust_zoom(self, factor: float) -> 'Viewport':
        return Viewport(position=self.position, zoom=self.zoom * factor)

class PygameWindow:
    """Handles pygame window initialization and basic operations."""
    def __init__(self, config: WindowConfig):
        pygame.init()
        self.config = config
        flags = pygame.RESIZABLE if config.resizable else 0
        self.screen = pygame.display.set_mode(
            (config.width, config.height),
            flags,
            vsync=1 if config.vsync else 0
        )
        pygame.display.set_caption(config.title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Viewport state
        self.viewport = Viewport.create()
        self.dragging = False
        self.last_mouse_pos = None
        
    @property
    def size(self) -> Tuple[int, int]:
        """Get current window size."""
        return self.screen.get_size()
    
    def world_to_screen(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to screen coordinates."""
        width, height = self.size
        screen_center = np.array([width/2, height/2])
        return (points + self.viewport.position) * self.viewport.zoom + screen_center
        
    def handle_input(self, event: pygame.event.Event) -> None:
        """Handle viewport-related input events."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                self.dragging = True
                self.last_mouse_pos = np.array(pygame.mouse.get_pos())
            elif event.button == 4:  # Mouse wheel up
                self.viewport = self.viewport.adjust_zoom(1.1)
            elif event.button == 5:  # Mouse wheel down
                self.viewport = self.viewport.adjust_zoom(1/1.1)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
                
        if self.dragging and self.last_mouse_pos is not None:
            current_mouse_pos = np.array(pygame.mouse.get_pos())
            delta = (current_mouse_pos - self.last_mouse_pos) / self.viewport.zoom
            self.viewport = self.viewport.move(-delta)
            self.last_mouse_pos = current_mouse_pos
        
    def render_text(self, text: str, pos: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        """Render text at given position."""
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, pos)
        
    def display_fps(self, pos: Tuple[int, int] = (10, 10)) -> None:
        """Display current FPS."""
        fps = self.clock.get_fps()
        self.render_text(f"FPS: {fps:.1f}", pos)
        
    def display_image(self, image: np.ndarray) -> None:
        """Display a numpy array as an image."""
        surface = pygame.surfarray.make_surface(image)
        self.screen.blit(surface, (0, 0))
        
    def update(self) -> None:
        """Update display."""
        pygame.display.flip()
        self.clock.tick()  # Let VSync control framerate

def run_renderer(
    window_config: WindowConfig,
    render_func: Callable[[np.ndarray, np.ndarray, np.ndarray, int, int], np.ndarray]
) -> None:
    """
    Run a renderer that takes world coordinates and produces images.
    
    Args:
        window_config: Window configuration
        render_func: Function that takes:
            - mouse_world_pos: np.ndarray[2] - Mouse position in world coordinates
            - view_center: np.ndarray[2] - View center in world coordinates
            - view_size: np.ndarray[2] - View size in world coordinates
            - width: int - Output width in pixels
            - height: int - Output height in pixels
            Returns: np.ndarray[height, width, 3] - RGB image
    """
    window = PygameWindow(window_config)
    running = True
    
    def render_frame(window: PygameWindow):
        width, height = window.size
        
        # Convert mouse position to world coordinates
        mouse_pos = np.array(pygame.mouse.get_pos())
        mouse_world_pos = (mouse_pos - np.array([width/2, height/2])) / window.viewport.zoom + window.viewport.position
        
        # Calculate view size in world coordinates
        view_size = np.array([width, height]) / window.viewport.zoom
        
        # Render
        image = render_func(
            mouse_world_pos,
            window.viewport.position,
            view_size,
            width,
            height
        )
        
        # Display
        window.display_image(image)
        window.display_fps()
        window.render_text("Click and drag to move, Scroll to zoom", (10, 50))
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            else:
                window.handle_input(event)
        
        render_frame(window)
        window.update()
    
    pygame.quit() 