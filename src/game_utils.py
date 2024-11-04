from dataclasses import dataclass
from typing import Protocol, Optional
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import glfw

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
    
    @staticmethod
    def create(config: WindowConfig) -> 'GameEngine':
        """Create initial engine state."""
        # Create canvas first
        canvas = WgpuCanvas(size=(config.width, config.height), title=config.title)
        engine = GameEngine(config=config, canvas=canvas)
        
        # Set up scroll callback using the canvas's window
        def scroll_callback(window, x_offset, y_offset):
            engine.scroll_offset = (x_offset, y_offset)
        
        glfw.set_scroll_callback(canvas._window, scroll_callback)
        return engine

    def run(self, initial_scene: SceneState) -> None:
        """Run the game loop."""
        scene = initial_scene
        last_time = glfw.get_time()
        
        def frame():
            nonlocal scene, last_time
            
            current_time = glfw.get_time()
            dt = current_time - last_time
            last_time = current_time
            
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
            
            # Request next frame
            self.canvas.request_draw(frame)
        
        # Start the event loop
        self.canvas.request_draw(frame)
        run()