import numpy as np
from dataclasses import dataclass, replace
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
from src.game_utils import SceneState, WindowConfig, GameEngine

class Gaussians(NamedTuple):
    """Represents a collection of 2D Gaussians."""
    pos: np.ndarray    # shape: (n, 2) for positions
    std: np.ndarray    # shape: (n,)
    intensity: np.ndarray  # shape: (n,)

def create_random_gaussians(n_points: int = 10000, spread: float = 500.0) -> Gaussians:
    """Create random gaussian points in world space."""
    rng = np.random.default_rng(0)
    return Gaussians(
        pos=rng.normal(0, spread, size=(n_points, 2)),
        std=np.exp(rng.normal(0, 1, size=n_points)) * 10.0,
        intensity=rng.uniform(0.2, 0.5, size=n_points)
    )

@dataclass(frozen=True)
class GameState(SceneState):
    """Complete game state."""
    view: ViewTransform
    gaussians: Gaussians
    device: wgpu.GPUDevice
    pipeline: wgpu.GPURenderPipeline
    bind_group: wgpu.GPUBindGroup
    vertex_buffer: wgpu.GPUBuffer
    instance_buffer: wgpu.GPUBuffer
    
    @staticmethod
    def create(width: int, height: int, canvas: WgpuCanvas) -> 'GameState':
        # Create WebGPU device
        adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
        device = adapter.request_device()
        
        # Create shader
        shader = device.create_shader_module(
            label="gaussian_shader",
            code="""
            struct VertexInput {
                @location(0) position: vec2f,
                @location(1) texcoord: vec2f,
                @location(2) instance_pos: vec2f,
                @location(3) instance_std: f32,
                @location(4) instance_intensity: f32,
            };

            struct ViewUniform {
                world_center: vec2f,
                world_size: vec2f,
            };
            @group(0) @binding(0) var<uniform> view: ViewUniform;

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) texcoord: vec2f,
                @location(1) std: f32,
                @location(2) intensity: f32,
            };

            @vertex
            fn vs_main(in: VertexInput) -> VertexOutput {
                var out: VertexOutput;
                
                // Transform instance position to screen space
                let screen_pos = (in.instance_pos - view.world_center) / (view.world_size * 0.5);
                
                // Scale quad by standard deviation
                let scaled_pos = screen_pos + in.position * (4.0 * in.instance_std / (view.world_size * 0.5));
                
                out.position = vec4f(scaled_pos, 0.0, 1.0);
                out.texcoord = in.texcoord;
                out.std = in.instance_std;
                out.intensity = in.instance_intensity;
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4f {
                // Compute gaussian value
                let sq_dist = dot(in.texcoord, in.texcoord);
                let value = in.intensity * exp(-0.5 * sq_dist);
                
                // Apply inferno colormap (simplified for now)
                let color = vec3f(value);  // Replace with proper inferno implementation
                return vec4f(color, value);
            }
            """
        )
        
        # Create pipeline
        pipeline = device.create_render_pipeline(
            label="gaussian_pipeline",
            layout=device.create_pipeline_layout(
                bind_group_layouts=[
                    device.create_bind_group_layout(
                        entries=[{
                            "binding": 0,
                            "visibility": wgpu.ShaderStage.VERTEX,
                            "buffer": {"type": "uniform"}
                        }]
                    )
                ]
            ),
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    # Vertex buffer layout
                    {
                        "array_stride": 16,
                        "attributes": [
                            {"format": "float32x2", "offset": 0, "shader_location": 0},  # position
                            {"format": "float32x2", "offset": 8, "shader_location": 1},  # texcoord
                        ]
                    },
                    # Instance buffer layout
                    {
                        "array_stride": 16,
                        "step_mode": "instance",
                        "attributes": [
                            {"format": "float32x2", "offset": 0, "shader_location": 2},  # instance_pos
                            {"format": "float32", "offset": 8, "shader_location": 3},    # instance_std
                            {"format": "float32", "offset": 12, "shader_location": 4},   # instance_intensity
                        ]
                    }
                ]
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": canvas.get_preferred_format()}]
            },
            primitive={
                "topology": "triangle-list",
                "front_face": "ccw",
                "cull_mode": "none"
            },
            depth_stencil=None,
            multisample={"count": 1}
        )
        
        # Create buffers and bind groups
        # ... (rest of initialization)
        
        return GameState(
            view=ViewTransform.create(width, height),
            gaussians=create_random_gaussians(),
            device=device,
            pipeline=pipeline,
            # ... other fields
        )
    
    def update(self, dt: float, window: int) -> Optional['GameState']:
        """Update game state."""
        mouse_pos = glfw.get_cursor_pos(window)
        new_view, changed = self.view.update(np.array(mouse_pos))
        return replace(self, view=new_view) if changed else self
    
    def handle_event(self, window: int, scroll_offset: tuple[float, float]) -> Optional['GameState']:
        """Handle input events."""
        new_view = self.view.handle_event(window, scroll_offset)
        if new_view is not None:
            return replace(self, view=new_view)
        return None
    
    def render(self) -> None:
        """Render all gaussians using instancing."""
        # Update uniforms
        self.program['u_world_center'].value = tuple(self.view.world_rect.center)
        self.program['u_world_size'].value = (self.view.world_rect.width, self.view.world_rect.height)
        
        # Set up blending
        self.ctx.clear()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Render
        self.vao.render(instances=len(self.gaussians.pos))
        
        # Clean up
        self.ctx.disable(moderngl.BLEND)

if __name__ == "__main__":
    config = WindowConfig(
        width=1024,
        height=768,
        title="Gaussian Game"
    )
    
    engine = GameEngine.create(config)
    game_state = GameState.create(config.width, config.height, engine.ctx)
    engine.run(game_state)