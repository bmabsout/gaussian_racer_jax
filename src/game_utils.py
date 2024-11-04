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
    def update(self, dt: float, window: int) -> Optional['SceneState']:
        """Update state and return new state if changed."""
        ...
    
    def handle_event(self, window: int, scroll_offset: tuple[float, float]) -> Optional['SceneState']:
        """Handle input and return new state if changed."""
        ...
    
    def render(self, canvas: WgpuCanvas) -> None:
        """Render current state."""
        ...

@dataclass
class GameEngine:
    config: WindowConfig
    canvas: WgpuCanvas
    scroll_offset: tuple[float, float] = (0.0, 0.0)
    fps_time: float = field(default_factory=time.time)
    frame_count: int = 0
    
    @staticmethod
    def create(config: WindowConfig) -> 'GameEngine':
        canvas = WgpuCanvas(
            size=(config.width, config.height),
            title=config.title,
            max_fps=240
        )
        
        engine = GameEngine(config=config, canvas=canvas)
        
        def scroll_callback(window, x_offset, y_offset):
            engine.scroll_offset = (x_offset, y_offset)
        
        glfw.set_scroll_callback(canvas._window, scroll_callback)
        return engine

    def run(self, initial_scene: SceneState) -> None:
        scene = initial_scene
        last_time = time.time()
        
        def frame():
            nonlocal scene, last_time
            current_time = time.time()
            
            # Update FPS counter
            self.frame_count += 1
            if current_time - self.fps_time >= 0.5:
                fps = self.frame_count / (current_time - self.fps_time)
                glfw.set_window_title(
                    self.canvas._window,
                    f"{self.config.title} - FPS: {fps:.1f}"
                )
                self.frame_count = 0
                self.fps_time = current_time
            
            # Update scene
            dt = current_time - last_time
            last_time = current_time
            
            glfw.poll_events()
            
            if new_scene := scene.handle_event(self.canvas._window, self.scroll_offset):
                scene = new_scene
            
            if new_scene := scene.update(dt, self.canvas._window):
                scene = new_scene
            
            self.scroll_offset = (0.0, 0.0)
            scene.render(self.canvas)
            
            self.canvas.request_draw(frame)
        
        self.canvas.request_draw(frame)
        run()