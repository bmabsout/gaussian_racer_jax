from dataclasses import dataclass
import numpy as np
from typing import Protocol
import pygame

@dataclass(frozen=True)
class WindowConfig:
    width: int
    height: int
    title: str = "Pygame Window"

class SceneState(Protocol):
    """Protocol for scene state."""
    def update(self, dt: float) -> 'SceneState':
        """Return new state after update."""
        ...
    
    def handle_event(self, event: pygame.event.Event) -> 'SceneState':
        """Return new state after handling event."""
        ...
    
    def render(self) -> np.ndarray:
        """Render current state to numpy array."""
        ...

@dataclass(frozen=True)
class GameEngine:
    """Functional game engine."""
    config: WindowConfig
    
    @staticmethod
    def create(config: WindowConfig) -> 'GameEngine':
        """Create initial engine state."""
        pygame.init()
        return GameEngine(config=config)

    def run(self, initial_scene: SceneState) -> None:
        """Run the game loop."""
        screen = pygame.display.set_mode(
            (self.config.width, self.config.height),
            pygame.RESIZABLE,
            vsync=1
        )
        pygame.display.set_caption(self.config.title)
        clock = pygame.time.Clock()
        
        scene = initial_scene
        last_time = pygame.time.get_ticks() / 1000.0
        
        while True:
            current_time = pygame.time.get_ticks() / 1000.0
            dt = current_time - last_time
            last_time = current_time
            
            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        pygame.quit()
                        return
                    case _:
                        scene = scene.handle_event(event)
            
            scene = scene.update(dt)
            
            image = scene.render()
            surface = pygame.surfarray.make_surface(image)
            screen.blit(surface, (0, 0))
            
            fps = 1.0 / dt if dt > 0 else 0
            font = pygame.font.Font(None, 36)
            fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
            screen.blit(fps_text, (10, 10))
            
            pygame.display.flip()
            clock.tick()