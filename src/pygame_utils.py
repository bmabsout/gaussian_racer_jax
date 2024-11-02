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
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, 
                                pygame.MOUSEBUTTONUP, pygame.WINDOWRESIZED])
        
        clock = pygame.time.Clock()
        scene = initial_scene
        running = True
        last_time = pygame.time.get_ticks() / 1000.0
        
        while running:
            current_time = pygame.time.get_ticks() / 1000.0
            dt = current_time - last_time
            last_time = current_time
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    scene = scene.handle_event(event)
            
            scene = scene.update(dt)
            scene.render()  # Direct rendering, no return value needed
            
            fps = clock.get_fps()
            # Render FPS using OpenGL text rendering or overlay
            
            pygame.display.flip()
            clock.tick()
        
        pygame.quit()