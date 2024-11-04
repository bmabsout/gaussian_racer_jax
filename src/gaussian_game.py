import numpy as np
from dataclasses import dataclass, replace
from typing import NamedTuple, Optional
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
from src.game_utils import SceneState, WindowConfig, GameEngine
from src.view_transform import ViewTransform
import glfw

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
        std=np.exp(rng.normal(0, 0.5, size=n_points)) * 20.0,
        intensity=rng.uniform(0.5, 1.0, size=n_points)
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
    view_uniform_buffer: wgpu.GPUBuffer
    
    @staticmethod
    def create(width: int, height: int, canvas: WgpuCanvas) -> 'GameState':
        # Create WebGPU device
        adapter = wgpu.gpu.request_adapter_sync(
            canvas=canvas,
            power_preference="high-performance"
        )
        device = adapter.request_device_sync()
        
        # Get context and configure it
        present_context = canvas.get_context()
        render_texture_format = present_context.get_preferred_format(adapter)
        present_context.configure(
            device=device,
            format=render_texture_format,
            alpha_mode="opaque",
            view_formats=[]
        )
        
        # Create shader program
        shader = device.create_shader_module(
            label="gaussian_shader",
            code="""
            struct VertexInput {
                @location(0) position: vec2f,
                @location(1) texcoord: vec2f,
                @location(2) instance_pos: vec2f,
                @location(3) instance_stddev: f32,
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
                @location(1) stddev: f32,
                @location(2) intensity: f32,
            };

            @vertex
            fn vs_main(in: VertexInput) -> VertexOutput {
                var out: VertexOutput;
                
                // Transform instance position to screen space
                let screen_pos = (in.instance_pos - view.world_center) / (view.world_size * 0.5);
                
                // Scale quad by standard deviation
                let scaled_pos = screen_pos + in.position * (4.0 * in.instance_stddev / (view.world_size * 0.5));
                
                out.position = vec4f(scaled_pos, 0.0, 1.0);
                out.texcoord = in.texcoord;
                out.stddev = in.instance_stddev;
                out.intensity = in.instance_intensity;
                return out;
            }

            fn inferno(t: f32) -> vec3f {
                // Inferno colormap coefficients
                let c0 = vec3f(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
                let c1 = vec3f(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
                let c2 = vec3f(11.60249308247187, -3.972853965665698, -15.9423941062914);
                let c3 = vec3f(-41.70399613139459, 17.43639888205313, 44.35414519872813);
                let c4 = vec3f(77.162935699427, -33.40235894210092, -81.80730925738993);
                let c5 = vec3f(-71.31942824499214, 32.62606426397723, 73.20951985803202);
                let c6 = vec3f(25.13112622477341, -12.24266895238567, -23.07032500287172);

                let t2 = t * t;
                let t3 = t2 * t;
                let t4 = t3 * t;
                let t5 = t4 * t;
                let t6 = t5 * t;

                return c0 + c1 * t + c2 * t2 + c3 * t3 + c4 * t4 + c5 * t5 + c6 * t6;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4f {
                // Compute gaussian value
                let sq_dist = dot(in.texcoord, in.texcoord);
                let std_sq = in.stddev * in.stddev;
                let value = in.intensity * exp(-0.5 * sq_dist);
                
                // Apply inferno colormap with proper alpha blending
                let color = inferno(clamp(value, 0.0, 1.0));
                return vec4f(color, value);  // Use value as alpha for proper blending
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
                            {"format": wgpu.VertexFormat.float32x2, "offset": 0, "shader_location": 0},  # position
                            {"format": wgpu.VertexFormat.float32x2, "offset": 8, "shader_location": 1},  # texcoord
                        ]
                    },
                    # Instance buffer layout
                    {
                        "array_stride": 16,
                        "step_mode": wgpu.VertexStepMode.instance,
                        "attributes": [
                            {"format": wgpu.VertexFormat.float32x2, "offset": 0, "shader_location": 2},  # instance_pos
                            {"format": wgpu.VertexFormat.float32, "offset": 8, "shader_location": 3},    # instance_stddev
                            {"format": wgpu.VertexFormat.float32, "offset": 12, "shader_location": 4},   # instance_intensity
                        ]
                    }
                ]
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{
                    "format": render_texture_format,
                    "blend": {
                        "color": {
                            "src_factor": wgpu.BlendFactor.src_alpha,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        },
                        "alpha": {
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        }
                    }
                }]
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none
            }
        )
        
        # Create vertex buffer with 6 vertices for 2 triangles
        vertices = np.array([
            # First triangle
            -4.0, -4.0,  -4.0, -4.0,  # pos, texcoord for vertex 0
             4.0, -4.0,   4.0, -4.0,  # pos, texcoord for vertex 1
             4.0,  4.0,   4.0,  4.0,  # pos, texcoord for vertex 2
            # Second triangle
            -4.0, -4.0,  -4.0, -4.0,  # pos, texcoord for vertex 0 again
             4.0,  4.0,   4.0,  4.0,  # pos, texcoord for vertex 2 again
            -4.0,  4.0,  -4.0,  4.0,  # pos, texcoord for vertex 3
        ], dtype=np.float32)
        
        vertex_buffer = device.create_buffer_with_data(
            data=vertices,
            usage=wgpu.BufferUsage.VERTEX
        )
        
        # Create instance data and buffer
        gaussians = create_random_gaussians()
        instance_data = np.zeros(len(gaussians.pos), dtype=[
            ('pos', 'f4', 2),
            ('std', 'f4', 1),
            ('intensity', 'f4', 1),
        ])
        instance_data['pos'] = gaussians.pos
        instance_data['std'] = gaussians.std
        instance_data['intensity'] = gaussians.intensity
        
        instance_buffer = device.create_buffer_with_data(
            data=instance_data,
            usage=wgpu.BufferUsage.VERTEX
        )
        
        # Create view uniform buffer
        view_uniform_buffer = device.create_buffer(
            size=16,  # vec2f world_center + vec2f world_size
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{
                "binding": 0,
                "resource": {"buffer": view_uniform_buffer}
            }]
        )
        
        return GameState(
            view=ViewTransform.create(width, height),
            gaussians=gaussians,
            device=device,
            pipeline=pipeline,
            bind_group=bind_group,
            vertex_buffer=vertex_buffer,
            instance_buffer=instance_buffer,
            view_uniform_buffer=view_uniform_buffer
        )
    
    def update(self, dt: float) -> Optional['GameState']:
        """Update game state."""
        return self

    def handle_event(self, scroll_offset: tuple[float, float]) -> Optional['GameState']:
        """Handle input events."""
        new_view = self.view.handle_event(scroll_offset)
        if new_view is not None:
            return replace(self, view=new_view)
        return None

    def render(self, canvas: WgpuCanvas) -> None:
        """Render the current state."""
        # Get current window size using GLFW
        width, height = glfw.get_window_size(canvas._window)
        
        # Reconfigure context if window size changed
        if (width, height) != tuple(self.view.screen_size):
            context = canvas.get_context()
            context.configure(
                device=self.device,
                format=context.get_preferred_format(self.device.adapter),
                alpha_mode="opaque",
                view_formats=[]
            )
            # Update view transform for new size
            new_view = self.view.handle_resize(width, height)
            self = replace(self, view=new_view)
        
        # Get the current texture view from the context
        try:
            current_texture = canvas.get_context().get_current_texture()
        except RuntimeError:
            # Skip frame if we can't get the texture
            return
            
        command_encoder = self.device.create_command_encoder()
        
        # Update view uniform buffer
        view_data = np.array([
            *self.view.world_rect.center,  # world_center (vec2f)
            self.view.world_rect.width,    # world_size.x (float)
            self.view.world_rect.height,   # world_size.y (float)
        ], dtype=np.float32)
        self.device.queue.write_buffer(
            self.view_uniform_buffer,
            0,
            view_data.tobytes()
        )
        
        # Create render pass
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": current_texture.create_view(),
                "clear_value": (0.0, 0.0, 0.0, 1.0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }]
        )
        
        # Set pipeline and bind groups
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        
        # Set vertex and instance buffers
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.set_vertex_buffer(1, self.instance_buffer)
        
        # Draw
        render_pass.draw(6, len(self.gaussians.pos))
        render_pass.end()
        
        # Submit commands
        self.device.queue.submit([command_encoder.finish()])

if __name__ == "__main__":
    config = WindowConfig(
        width=1024,
        height=768,
        title="Gaussian Game"
    )
    
    engine = GameEngine.create(config)
    game_state = GameState.create(config.width, config.height, engine.canvas)
    engine.run(game_state)