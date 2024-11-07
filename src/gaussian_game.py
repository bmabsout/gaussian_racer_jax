from dataclasses import dataclass, replace
from typing import Optional
import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas
import glfw
import time

from src.game_engine import SceneState, WindowConfig, GameEngine
from src.view_transform import ViewTransform
from src.gaussians import Gaussians, create_random_gaussians
from src.webgpu.pipeline import create_pipelines, create_buffers

@dataclass(frozen=True)
class GameState(SceneState):
    view: ViewTransform
    gaussians: Gaussians
    device: wgpu.GPUDevice
    accumulation_pipeline: wgpu.GPURenderPipeline
    colormap_pipeline: wgpu.GPURenderPipeline
    bind_group: wgpu.GPUBindGroup
    gaussian_vertex_buffer: wgpu.GPUBuffer
    fullscreen_vertex_buffer: wgpu.GPUBuffer
    instance_buffer: wgpu.GPUBuffer
    view_uniform_buffer: wgpu.GPUBuffer
    colormap_bind_group_layout: wgpu.GPUBindGroupLayout
    mouse_pos: Optional[np.ndarray] = None
    last_placed_pos: Optional[np.ndarray] = None  # Track last placed gaussian position

    @staticmethod
    def create(width: int, height: int, canvas: WgpuCanvas, device: wgpu.GPUDevice) -> 'GameState':
        # Configure canvas
        context = canvas.get_context()
        format = context.get_preferred_format(device.adapter)
        context.configure(
            device=device,
            format=format,
            alpha_mode="opaque",
        )
        
        # Create gaussians
        gaussians = create_random_gaussians()
        
        # Create pipelines and buffers
        accumulation_pipeline, colormap_pipeline, view_bind_group_layout, colormap_bind_group_layout = create_pipelines(device, format)
        gaussian_vertex_buffer, fullscreen_vertex_buffer, instance_buffer, view_uniform_buffer = create_buffers(device, gaussians)
        
        # Create view bind group
        bind_group = device.create_bind_group(
            layout=view_bind_group_layout,
            entries=[{
                "binding": 0,
                "resource": {"buffer": view_uniform_buffer}
            }]
        )

        return GameState(
            view=ViewTransform.create(width, height),
            gaussians=gaussians,
            device=device,
            accumulation_pipeline=accumulation_pipeline,
            colormap_pipeline=colormap_pipeline,
            bind_group=bind_group,
            gaussian_vertex_buffer=gaussian_vertex_buffer,
            fullscreen_vertex_buffer=fullscreen_vertex_buffer,
            instance_buffer=instance_buffer,
            view_uniform_buffer=view_uniform_buffer,
            colormap_bind_group_layout=colormap_bind_group_layout,
            mouse_pos=None,
            last_placed_pos=None  # Initialize last placed position
        )

    def handle_event(self, window: int, scroll_offset: tuple[float, float]) -> Optional['GameState']:
        # Handle resize first
        width, height = glfw.get_window_size(window)
        if (width, height) != (self.view.screen_rect.width, self.view.screen_rect.height):
            new_view = self.view.handle_resize(width, height)
            # Update mouse position with new view
            if self.mouse_pos is not None:
                mouse_screen_pos = np.array(glfw.get_cursor_pos(window))
                mouse_world_pos = new_view.screen_to_world(mouse_screen_pos)
                return replace(self, view=new_view, mouse_pos=mouse_world_pos)
            return replace(self, view=new_view)

        # Get mouse position and shift state
        mouse_screen_pos = np.array(glfw.get_cursor_pos(window))
        mouse_world_pos = self.view.screen_to_world(mouse_screen_pos)
        left_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        shift_pressed = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
                       glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        # Add gaussian when shift is pressed and moved enough
        std = 20.0
        if shift_pressed:
            should_place = False
            if self.last_placed_pos is None:
                should_place = True
            else:
                distance = np.linalg.norm(mouse_world_pos - self.last_placed_pos)
                should_place = distance > std/2

            if should_place:
                new_gaussians = Gaussians(
                    pos=np.append(self.gaussians.pos, [mouse_world_pos], axis=0),
                    std=np.append(self.gaussians.std, [std]),
                    intensity=np.append(self.gaussians.intensity, [1.0])
                )
                
                # Create new buffer if needed
                if (len(new_gaussians.pos) + 1) * 16 > self.instance_buffer.size:
                    new_size = (len(new_gaussians.pos) + 1000) * 16  # Add extra space
                    new_buffer = self.device.create_buffer(
                        size=new_size,
                        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
                    )
                    return replace(self,
                                 gaussians=new_gaussians,
                                 instance_buffer=new_buffer,
                                 last_placed_pos=mouse_world_pos)
                
                return replace(self,
                             gaussians=new_gaussians,
                             last_placed_pos=mouse_world_pos)

        # Update view transform
        new_view = self.view.handle_event(scroll_offset, mouse_screen_pos, left_pressed)
        if new_view is not None:
            return replace(self, view=new_view)

        # Update mouse position
        if mouse_world_pos is not None and not np.array_equal(mouse_world_pos, self.mouse_pos):
            return replace(self, mouse_pos=mouse_world_pos)

        return None

    def render(self, canvas: WgpuCanvas) -> None:
        start_time = time.time()
        
        # Handle resize and update mouse position
        width, height = glfw.get_window_size(canvas._window)
        if (width, height) != (self.view.screen_rect.width, self.view.screen_rect.height):
            new_view = self.view.handle_resize(width, height)
            # Update mouse position with new view
            if self.mouse_pos is not None:
                mouse_screen_pos = np.array(glfw.get_cursor_pos(canvas._window))
                mouse_world_pos = new_view.screen_to_world(mouse_screen_pos)
                self = replace(self, view=new_view, mouse_pos=mouse_world_pos)
            else:
                self = replace(self, view=new_view)
        
        try:
            current_texture = canvas.get_context().get_current_texture()
        except RuntimeError as e:
            if "Cannot get surface texture (2)" in str(e):
                return
            raise
            
        # Create intermediate texture for accumulation
        width, height = current_texture.width, current_texture.height
        accumulation_texture = self.device.create_texture(
            size={"width": width, "height": height, "depth_or_array_layers": 1},
            format=wgpu.TextureFormat.r16float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
        )
        
        # Create sampler for colormap pass
        sampler = self.device.create_sampler(
            min_filter="linear",
            mag_filter="linear",
            mipmap_filter="linear",
        )
        
        # Create bind group for colormap pass
        colormap_bind_group = self.device.create_bind_group(
            layout=self.colormap_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": accumulation_texture.create_view()
                },
                {
                    "binding": 1,
                    "resource": sampler
                }
            ]
        )
            
        # Update instance data
        instance_data = np.zeros(len(self.gaussians.pos) + 1, dtype=np.dtype([
            ('pos', np.float32, 2),
            ('std', np.float32, 1),
            ('intensity', np.float32, 1),
        ]))
        instance_data['pos'][:len(self.gaussians.pos)] = self.gaussians.pos
        instance_data['std'][:len(self.gaussians.pos)] = self.gaussians.std
        instance_data['intensity'][:len(self.gaussians.pos)] = self.gaussians.intensity
        
        if self.mouse_pos is not None:
            instance_data['pos'][-1] = self.mouse_pos
            instance_data['std'][-1] = 20.0
            instance_data['intensity'][-1] = 1.0
        
        buffer_start = time.time()
        self.device.queue.write_buffer(self.instance_buffer, 0, instance_data.tobytes())
        buffer_end = time.time()
        
        print(f"Buffer update took: {(buffer_end - buffer_start)*1000:.2f}ms")
        print(f"Buffer size: {instance_data.nbytes/1024:.2f}KB")
        
        # Update view uniforms
        view_data = np.array([
            *self.view.world_rect.center,
            self.view.world_rect.width,
            self.view.world_rect.height,
        ], dtype=np.float32)
        self.device.queue.write_buffer(self.view_uniform_buffer, 0, view_data.tobytes())
        
        command_encoder = self.device.create_command_encoder()
        
        # First pass: accumulate gaussians to intermediate texture
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": accumulation_texture.create_view(),
                "clear_value": (0.0, 0.0, 0.0, 1.0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }]
        )
        
        render_pass.set_pipeline(self.accumulation_pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.set_vertex_buffer(0, self.gaussian_vertex_buffer)
        render_pass.set_vertex_buffer(1, self.instance_buffer)
        render_pass.draw(6, len(instance_data))
        render_pass.end()
        
        # Second pass: apply colormap to final texture
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": current_texture.create_view(),
                "clear_value": (0.0, 0.0, 0.0, 1.0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }]
        )
        
        render_pass.set_pipeline(self.colormap_pipeline)
        render_pass.set_bind_group(0, colormap_bind_group)
        render_pass.set_vertex_buffer(0, self.fullscreen_vertex_buffer)
        render_pass.draw(3, 1)
        render_pass.end()
        
        self.device.queue.submit([command_encoder.finish()])

if __name__ == "__main__":
    from src.game_engine import WindowConfig, GameEngine
    
    config = WindowConfig(
        width=1024,
        height=768,
        title="Gaussian Game"
    )
    
    engine = GameEngine.create(config)
    game_state = GameState.create(config.width, config.height, engine.canvas, engine.device)
    engine.run(game_state)