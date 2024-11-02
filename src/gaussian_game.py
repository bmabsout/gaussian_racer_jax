import numpy as np
from dataclasses import dataclass, replace
import glfw
import moderngl
from src.game_utils import SceneState, WindowConfig, GameEngine
from src.view_transform import ViewTransform
from typing import Optional, NamedTuple

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
    ctx: moderngl.Context
    program: moderngl.Program
    vao: moderngl.VertexArray
    instance_buffer: moderngl.Buffer  # New: for instanced rendering
    
    @staticmethod
    def create(width: int, height: int, ctx: moderngl.Context) -> 'GameState':
        program = ctx.program(
            vertex_shader='''
                #version 430
                
                // Quad vertices
                in vec2 in_position;
                in vec2 in_texcoord;
                
                // Instance data
                in vec2 in_instance_pos;
                in float in_instance_std;
                in float in_instance_intensity;
                
                // View uniforms
                uniform vec2 u_world_center;
                uniform vec2 u_world_size;
                
                out vec2 v_texcoord;
                out float v_std;
                out float v_intensity;
                
                void main() {
                    // Pass through instance data
                    v_std = in_instance_std;
                    v_intensity = in_instance_intensity;
                    v_texcoord = in_texcoord;
                    
                    // Transform instance position to screen space
                    vec2 screen_pos = (in_instance_pos - u_world_center) / (u_world_size * 0.5);
                    
                    // Scale quad by standard deviation (in screen space)
                    // Using 8 standard deviations for very wide coverage
                    vec2 scaled_pos = screen_pos + in_position * (8.0 * in_instance_std / (u_world_size * 0.5));
                    
                    gl_Position = vec4(scaled_pos, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 430
                
                in vec2 v_texcoord;
                in float v_std;
                in float v_intensity;
                
                out vec4 f_color;
                
                vec3 inferno(float t) {
                    const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
                    const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
                    const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
                    const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
                    const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
                    const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
                    const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);
                    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
                }
                
                void main() {
                    // Compute gaussian value
                    float sq_dist = dot(v_texcoord, v_texcoord);
                    float value = v_intensity * exp(-0.5 * sq_dist);
                    
                    // Apply colormap
                    vec3 color = inferno(clamp(value, 0.0, 1.0));
                    f_color = vec4(color, value);  // Use value as alpha for proper blending
                }
            '''
        )
        
        # Create quad vertices (made much larger for better coverage)
        vertices = np.array([
            -4.0, -4.0,  -4.0, -4.0,  # Quadrupled the size of the quad
             4.0, -4.0,   4.0, -4.0,
             4.0,  4.0,   4.0,  4.0,
            -4.0,  4.0,  -4.0,  4.0,
        ], dtype='f4')
        
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='i4')
        
        vbo = ctx.buffer(vertices.tobytes())
        ibo = ctx.buffer(indices.tobytes())
        
        # Create instance data
        gaussians = create_random_gaussians()
        instance_data = np.zeros(len(gaussians.pos), dtype=[
            ('pos', 'f4', 2),
            ('std', 'f4', 1),
            ('intensity', 'f4', 1),
        ])
        instance_data['pos'] = gaussians.pos
        instance_data['std'] = gaussians.std
        instance_data['intensity'] = gaussians.intensity
        instance_buffer = ctx.buffer(instance_data.tobytes())
        
        vao = ctx.vertex_array(
            program,
            [
                (vbo, '2f 2f', 'in_position', 'in_texcoord'),
                (instance_buffer, '2f 1f 1f/i', 'in_instance_pos', 'in_instance_std', 'in_instance_intensity'),
            ],
            ibo
        )
        
        return GameState(
            view=ViewTransform.create(width, height),
            gaussians=gaussians,
            ctx=ctx,
            program=program,
            vao=vao,
            instance_buffer=instance_buffer
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