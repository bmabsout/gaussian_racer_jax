import wgpu
import numpy as np
from .gaussian_shader import GAUSSIAN_SHADER
from .colormap_shader import generate_colormap_shader_code
from ..gaussians import Gaussians

def create_pipelines(device: wgpu.GPUDevice, surface_format: wgpu.TextureFormat):
    # Create shaders
    accumulation_shader = device.create_shader_module(
        label="accumulation_shader",
        code=GAUSSIAN_SHADER
    )
    
    colormap_shader = device.create_shader_module(
        label="colormap_shader",
        code=generate_colormap_shader_code()
    )
    
    # Create pipeline layouts
    view_bind_group_layout = device.create_bind_group_layout(
        entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.VERTEX,
            "buffer": {"type": "uniform"}
        }]
    )
    
    colormap_bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2
                }
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering}
            }
        ]
    )
    
    # Create accumulation pipeline (renders to r16float texture)
    accumulation_pipeline = device.create_render_pipeline(
        label="accumulation_pipeline",
        layout=device.create_pipeline_layout(
            bind_group_layouts=[view_bind_group_layout]
        ),
        vertex={
            "module": accumulation_shader,
            "entry_point": "vs_main",
            "buffers": [
                {  # Vertex buffer
                    "array_stride": 16,
                    "attributes": [
                        {"format": wgpu.VertexFormat.float32x2, "offset": 0, "shader_location": 0},
                        {"format": wgpu.VertexFormat.float32x2, "offset": 8, "shader_location": 1}
                    ]
                },
                {  # Instance buffer
                    "array_stride": 16,
                    "step_mode": wgpu.VertexStepMode.instance,
                    "attributes": [
                        {"format": wgpu.VertexFormat.float32x2, "offset": 0, "shader_location": 2},
                        {"format": wgpu.VertexFormat.float32, "offset": 8, "shader_location": 3},
                        {"format": wgpu.VertexFormat.float32, "offset": 12, "shader_location": 4}
                    ]
                }
            ]
        },
        fragment={
            "module": accumulation_shader,
            "entry_point": "fs_main",
            "targets": [{
                "format": wgpu.TextureFormat.r16float,  # Single 16-bit float for height
                "blend": {
                    "color": {
                        "src_factor": wgpu.BlendFactor.one,
                        "dst_factor": wgpu.BlendFactor.one,
                        "operation": wgpu.BlendOperation.add
                    },
                    "alpha": {
                        "src_factor": wgpu.BlendFactor.one,
                        "dst_factor": wgpu.BlendFactor.one,
                        "operation": wgpu.BlendOperation.add
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
    
    # Create colormap pipeline (renders to surface format)
    colormap_pipeline = device.create_render_pipeline(
        label="colormap_pipeline",
        layout=device.create_pipeline_layout(
            bind_group_layouts=[colormap_bind_group_layout]
        ),
        vertex={
            "module": colormap_shader,
            "entry_point": "vs_main",
            "buffers": [{
                "array_stride": 16,
                "attributes": [
                    {"format": wgpu.VertexFormat.float32x2, "offset": 0, "shader_location": 0},
                    {"format": wgpu.VertexFormat.float32x2, "offset": 8, "shader_location": 1}
                ]
            }]
        },
        fragment={
            "module": colormap_shader,
            "entry_point": "fs_main",
            "targets": [{
                "format": surface_format
            }]
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none
        }
    )
    
    return accumulation_pipeline, colormap_pipeline, view_bind_group_layout, colormap_bind_group_layout

def create_buffers(device: wgpu.GPUDevice, gaussians: Gaussians):
    # Create vertex buffers
    gaussian_vertices = np.array([
        # First triangle
        -4.0, -4.0,  -4.0, -4.0,
         4.0, -4.0,   4.0, -4.0,
         4.0,  4.0,   4.0,  4.0,
        # Second triangle
        -4.0, -4.0,  -4.0, -4.0,
         4.0,  4.0,   4.0,  4.0,
        -4.0,  4.0,  -4.0,  4.0,
    ], dtype=np.float32)
    
    fullscreen_vertices = np.array([
        # First triangle (fullscreen quad)
        -1.0, -1.0,   0.0, 1.0,
         3.0, -1.0,   2.0, 1.0,
        -1.0,  3.0,   0.0, -1.0,
    ], dtype=np.float32)
    
    # Create instance data with fixed size
    instance_data = np.zeros(len(gaussians.pos) + 1000, dtype=np.dtype([  # Add extra space
        ('pos', np.float32, 2),
        ('std', np.float32, 1),
        ('intensity', np.float32, 1),
    ]))
    instance_data['pos'][:len(gaussians.pos)] = gaussians.pos
    instance_data['std'][:len(gaussians.pos)] = gaussians.std
    instance_data['intensity'][:len(gaussians.pos)] = gaussians.intensity
    
    gaussian_vertex_buffer = device.create_buffer_with_data(
        data=gaussian_vertices,
        usage=wgpu.BufferUsage.VERTEX
    )
    
    fullscreen_vertex_buffer = device.create_buffer_with_data(
        data=fullscreen_vertices,
        usage=wgpu.BufferUsage.VERTEX
    )
    
    instance_buffer = device.create_buffer_with_data(
        data=instance_data,
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
    )
    
    view_uniform_buffer = device.create_buffer(
        size=16,  # vec2f world_center + vec2f world_size
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    
    return gaussian_vertex_buffer, fullscreen_vertex_buffer, instance_buffer, view_uniform_buffer