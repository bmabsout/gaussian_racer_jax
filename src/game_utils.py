from dataclasses import dataclass, field
from typing import Protocol, Optional, Callable
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
    fps_update_interval: float = 0.5
    last_fps_update: float = field(default_factory=time.time)
    current_fps: float = 0.0
    running: bool = True
    
    @staticmethod
    def create(config: WindowConfig) -> 'GameEngine':
        """Create initial engine state."""
        canvas = WgpuCanvas(
            size=(config.width, config.height),
            title=config.title,
            max_fps=240  # Try to disable frame limiting
        )
        
        engine = GameEngine(config=config, canvas=canvas)
        
        def scroll_callback(window, x_offset, y_offset):
            engine.scroll_offset = (x_offset, y_offset)
        
        glfw.set_scroll_callback(canvas._window, scroll_callback)
        
        # Handle window close
        def close_callback(window):
            engine.running = False
        
        glfw.set_window_close_callback(canvas._window, close_callback)
        
        return engine

    def run(self, initial_scene: SceneState) -> None:
        """Run the game loop."""
        scene = initial_scene
        
        def frame():
            nonlocal scene
            
            if not self.running:
                return
            
            current_time = time.time()
            
            # Update FPS
            self.frame_count += 1
            if current_time - self.last_fps_update >= self.fps_update_interval:
                self.current_fps = self.frame_count / (current_time - self.last_fps_update)
                self.frame_count = 0
                self.last_fps_update = current_time
                glfw.set_window_title(
                    self.canvas._window,
                    f"{self.config.title} - FPS: {self.current_fps:.1f}"
                )
            
            # Calculate delta time
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # Poll events
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
            
            # Request next frame immediately if still running
            if self.running:
                self.canvas.request_draw(frame)
        
        # Start the event loop
        self.canvas.request_draw(frame)
        run()