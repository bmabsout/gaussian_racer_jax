import jax
import jax.numpy as jnp
from dataclasses import dataclass, replace
import glfw
import moderngl
import numpy as np
from src.game_utils import SceneState, WindowConfig, GameEngine
from src.gaussian_utils import apply_colormap
from src.simple_gaussian_renderer import Gaussians
from src.view_transform import ViewTransform


def create_random_gaussians(n_points: int = 2000, spread: float = 500.0) -> Gaussians:
    """Create random gaussian points in world space."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    
    return Gaussians(
        pos=jax.random.normal(k1, shape=(n_points, 2)) * spread,
        std=jnp.exp(jax.random.normal(k2, shape=(n_points,))) * 10.0,
        intensity=jax.random.uniform(k3, shape=(n_points,), minval=0.2, maxval=0.5)
    )

@dataclass(frozen=True)
class GameState(SceneState):
    """Complete game state."""
    view: ViewTransform
    gaussians: Gaussians
    ctx: moderngl.Context
    texture: moderngl.Texture
    program: moderngl.Program
    vao: moderngl.VertexArray
    
    @staticmethod
    def create(width: int, height: int, ctx: moderngl.Context) -> 'GameState':
        # Create shader program
        program = ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    v_texcoord = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D texture0;
                in vec2 v_texcoord;
                out vec4 f_color;
                void main() {
                    float value = texture(texture0, v_texcoord).r;
                    f_color = vec4(value, value, value, 1.0);
                }
            '''
        )
        
        # Create fullscreen quad
        vertices = np.array([
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0,
        ], dtype='f4')
        
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='i4')
        
        vbo = ctx.buffer(vertices.tobytes())
        ibo = ctx.buffer(indices.tobytes())
        
        vao = ctx.vertex_array(
            program,
            [(vbo, '2f 2f', 'in_position', 'in_texcoord')],
            ibo
        )
        
        # Create texture
        texture = ctx.texture((width, height), 1, dtype='f4')
        
        return GameState(
            view=ViewTransform.create(width, height),
            gaussians=create_random_gaussians(),
            ctx=ctx,
            texture=texture,
            program=program,
            vao=vao
        )
    
    def update(self, dt: float, window: int) -> 'GameState':
        """Update game state."""
        mouse_pos = glfw.get_cursor_pos(window)
        new_view, changed = self.view.update(jnp.array(mouse_pos))
        return replace(self, view=new_view) if changed else self
    
    def handle_event(self, window: int) -> 'GameState':
        """Handle input events."""
        width, height = glfw.get_window_size(window)
        if (width, height) != tuple(self.view.screen_size):
            new_view = self.view.handle_resize(width, height)
            new_texture = self.ctx.texture((width, height), 1, dtype='f4')
            return replace(self, view=new_view, texture=new_texture)
        return self
    
    def render(self) -> None:
        """Render the current state."""
        # Render gaussians (stays on GPU)
        values = self.view.positions[:,:,0]*0
        
        # Update texture
        self.texture.use(0)
        self.program['texture0'].value = 0
        
        # Clear and render
        self.ctx.clear()
        self.vao.render()

if __name__ == "__main__":
    config = WindowConfig(
        width=1024,
        height=768,
        title="Gaussian Game"
    )
    
    engine = GameEngine.create(config)
    game_state = GameState.create(config.width, config.height, engine.ctx)
    engine.run(game_state)