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
    def update(self, dt: float, window: int) -> Optional['SceneState']:
        """Return new state after update."""
        ...
    
    def handle_event(self, window: int, scroll_offset: tuple[float, float]) -> Optional['SceneState']:
        """Handle input events and return new state."""
        ...
    
    def render(self) -> None:
        """Render current state directly to screen."""
        ...

@dataclass
class GameEngine:
    """Functional game engine using GLFW."""
    config: WindowConfig
    window: int
    ctx: moderngl.Context
    scroll_offset: tuple[float, float] = (0.0, 0.0)
    
    @staticmethod
    def create(config: WindowConfig) -> 'GameEngine':
        """Create initial engine state."""
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
        # Request a modern OpenGL context that works across platforms
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
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
        
        engine = GameEngine(config=config, window=window, ctx=ctx)
        
        # Set up scroll callback
        def scroll_callback(window, x_offset, y_offset):
            engine.scroll_offset = (x_offset, y_offset)
        
        glfw.set_scroll_callback(window, scroll_callback)
        
        return engine

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
            
            # Handle events
            new_scene = scene.handle_event(self.window, self.scroll_offset)
            if new_scene is not None:
                scene = new_scene
            
            # Update scene
            new_scene = scene.update(dt, self.window)
            if new_scene is not None:
                scene = new_scene
            
            # Reset scroll offset after handling events
            self.scroll_offset = (0.0, 0.0)
            
            # Render
            scene.render()
            glfw.swap_buffers(self.window)
            
            if dt > 0:
                fps = 1.0 / dt
                glfw.set_window_title(self.window, f"{self.config.title} - FPS: {fps:.1f}")
        
        glfw.terminate()