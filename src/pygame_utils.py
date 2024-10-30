from dataclasses import dataclass
import numpy as np
from typing import Protocol, Optional
import pygame

@dataclass
class WindowConfig:
    width: int
    height: int
    title: str = "Pygame Window"
    resizable: bool = True
    vsync: bool = True

class Scene(Protocol):
    """Protocol for scenes to implement."""
    def update(self, dt: float) -> None:
        """Update scene state."""
        ...
    
    def render(self) -> np.ndarray:
        """Render scene to numpy array."""
        ...
    
    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle pygame events."""
        ...

class GameEngine:
    """Core game engine handling window and scene management."""
    def __init__(self, config: WindowConfig):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config.width, config.height),
            pygame.RESIZABLE if config.resizable else 0,
            vsync=1 if config.vsync else 0
        )
        pygame.display.set_caption(config.title)
        self.clock = pygame.time.Clock()
        self.current_scene: Optional[Scene] = None
    
    def run(self, scene: Scene) -> None:
        """Run the game with given scene."""
        self.current_scene = scene
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    self.current_scene.handle_event(event)
            
            # Update
            dt = self.clock.tick() / 1000.0  # Convert to seconds
            self.current_scene.update(dt)
            
            # Render
            image = self.current_scene.render()
            surface = pygame.surfarray.make_surface(image)
            self.screen.blit(surface, (0, 0))
            
            # Show FPS
            font = pygame.font.Font(None, 36)
            fps_text = font.render(f"FPS: {self.clock.get_fps():.1f}", True, (255, 255, 255))
            self.screen.blit(fps_text, (10, 10))
            
            pygame.display.flip()
        
        pygame.quit() 