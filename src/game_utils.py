from dataclasses import dataclass, field
from typing import Protocol, Optional
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import glfw
import time

@dataclass(frozen=True)
class WindowConfig:
    width: int
    height: int
    title: str = "WebGPU Window"

class SceneState(Protocol):
    """Protocol for scene state."""
    def update(self, dt: float) -> Optional['SceneState']:
        """Return new state after update."""
        ...
    
    def handle_event(self, scroll_offset: tuple[float, float]) -> Optional['SceneState']:
        """Handle input events and return new state."""
        ...
    
    def render(self, canvas: WgpuCanvas) -> None:
        """Render current state directly to screen."""
        ...

@dataclass
class GameEngine:
    """Game engine using WebGPU."""
    config: WindowConfig
    canvas: WgpuCanvas
    scroll_offset: tuple[float, float] = (0.0, 0.0)
    last_time: float = field(default_factory=time.time)
    frame_count: int = 0
    fps_update_interval: float = 0.5  # Update FPS every half second
    last_fps_update: float = field(default_factory=time.time)
    current_fps: float = 0.0
    
    @staticmethod
    def create(config: WindowConfig) -> 'GameEngine':
        """Create initial engine state."""
        # Initialize GLFW window with correct parameters
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
            
        # Set up window parameters with vsync disabled
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.swap_interval(0)  # Disable vsync
        
        # Create GLFW window
        window = glfw.create_window(config.width, config.height, config.title, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Could not create window")
            
        # Create canvas with existing window
        canvas = WgpuCanvas(window=window)
        
        engine = GameEngine(config=config, canvas=canvas)
        
        # Set up scroll callback
        def scroll_callback(window, x_offset, y_offset):
            engine.scroll_offset = (x_offset, y_offset)
        
        glfw.set_scroll_callback(window, scroll_callback)
        return engine

    def run(self, initial_scene: SceneState) -> None:
        """Run the game loop."""
        scene = initial_scene
        
        def frame():
            nonlocal scene
            current_time = time.time()
            
            # Update FPS
            self.frame_count += 1
            if current_time - self.last_fps_update >= self.fps_update_interval:
                self.current_fps = self.frame_count / (current_time - self.last_fps_update)
                self.frame_count = 0
                self.last_fps_update = current_time
                # Update window title with FPS
                glfw.set_window_title(
                    self.canvas._window,
                    f"{self.config.title} - FPS: {self.current_fps:.1f}"
                )
            
            # Calculate delta time
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # Poll GLFW events
            glfw.poll_events()
            
            # Handle events
            new_scene = scene.handle_event(self.scroll_offset)
            if new_scene is not None:
                scene = new_scene
            
            # Update scene
            new_scene = scene.update(dt)
            if new_scene is not None:
                scene = new_scene
            
            # Reset scroll offset
            self.scroll_offset = (0.0, 0.0)
            
            # Render
            scene.render(self.canvas)
            
            # Request next frame immediately
            self.canvas.request_draw(frame)
        
        # Start the event loop
        self.canvas.request_draw(frame)
        run()