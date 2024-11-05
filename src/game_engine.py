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
        """Return new state after update."""
        ...
    
    def handle_event(self, window: int, scroll_offset: tuple[float, float]) -> Optional['SceneState']:
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
    device: wgpu.GPUDevice
    adapter: wgpu.GPUAdapter
    scroll_offset: tuple[float, float] = (0.0, 0.0)
    last_time: float = field(default_factory=time.time)
    frame_count: int = 0
    fps_update_interval: float = 0.5
    last_fps_update: float = field(default_factory=time.time)
    current_fps: float = 0.0
    
    @staticmethod
    def create(config: WindowConfig) -> 'GameEngine':
        canvas = WgpuCanvas(
            size=(config.width, config.height),
            title=config.title,
            max_fps=240
        )
        
        # Create device
        adapter = wgpu.gpu.request_adapter_sync(
            canvas=canvas,
            power_preference="high-performance"
        )
        device = adapter.request_device_sync()
        
        # Initial context configuration
        context = canvas.get_context()
        context.configure(
            device=device,
            format=context.get_preferred_format(adapter),
            alpha_mode="opaque",
            view_formats=[]
        )
        
        engine = GameEngine(
            config=config,
            canvas=canvas,
            device=device,
            adapter=adapter
        )
        
        def scroll_callback(window, x_offset, y_offset):
            engine.scroll_offset = (x_offset, y_offset)
        
        glfw.set_scroll_callback(canvas._window, scroll_callback)
        return engine

    def run(self, initial_scene: SceneState) -> None:
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
            new_scene = scene.handle_event(self.canvas._window, self.scroll_offset)
            if new_scene is not None:
                scene = new_scene
            
            # Update scene
            new_scene = scene.update(dt, self.canvas._window)
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