import numpy as np
import matplotlib.cm as cm

# Pre-compute colormap
COLORMAP = (cm.get_cmap('inferno')(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

def apply_colormap(image: np.ndarray) -> np.ndarray:
    """Apply the inferno colormap to an image."""
    return COLORMAP[(image * 255).astype(np.uint8)]