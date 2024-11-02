from dataclasses import dataclass
from typing import Protocol, Optional
import glfw
import moderngl

@dataclass(frozen=True)
class WindowConfig:
    width: int
    height: int
    title: str = "GLFW Window"

class SceneState(Protocol):
    """Protocol for scene state."""
    def update(self, dt: float, window: int) -> 'SceneState':
        """Return new state after update."""
        ...
    
    def handle_event(self, window: int) -> 'SceneState':
        """Handle input events and return new state."""
        ...
    
    def render(self) -> None:
        """Render current state directly to screen."""
        ...
    
    @staticmethod
    def create(width: int, height: int, ctx: moderngl.Context) -> 'SceneState':
        """Create initial state with given context."""
        ...

@dataclass(frozen=True)
class GameEngine:
    """Functional game engine using GLFW."""
    config: WindowConfig
    window: int
    ctx: moderngl.Context
    
    @staticmethod
    def create(config: WindowConfig) -> 'GameEngine':
        """Create initial engine state."""
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
            
        # Configure GLFW window hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        
        # Create window
        window = glfw.create_window(
            config.width,
            config.height,
            config.title,
            None,
            None
        )
        
        if not window:
            glfw.terminate()
            raise RuntimeError("Could not create window")
            
        glfw.make_context_current(window)
        
        # Create ModernGL context
        ctx = moderngl.create_context()
        
        return GameEngine(config=config, window=window, ctx=ctx)

    def run(self, initial_scene: SceneState) -> None:
        """Run the game loop."""
        scene = initial_scene
        last_time = glfw.get_time()
        
        while not glfw.window_should_close(self.window):
            current_time = glfw.get_time()
            dt = current_time - last_time
            last_time = current_time
            
            # Poll events and update scene
            glfw.poll_events()
            scene = scene.handle_event(self.window)
            scene = scene.update(dt, self.window)
            
            # Render
            scene.render()
            glfw.swap_buffers(self.window)
            
            # Calculate and display FPS
            if dt > 0:
                fps = 1.0 / dt
                glfw.set_window_title(self.window, f"{self.config.title} - FPS: {fps:.1f}")
        
        glfw.terminate() 