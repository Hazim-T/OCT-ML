import matplotlib.pyplot as plt
import eyepy as ep
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def normalize(data):
    return data / 255.0
    #return (data - data.min()) / (data.max() - data.min())

def show_scan(scan, x=5, y=5):
    fig, axes = plt.subplots(x, y, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(scan.shape[0]):
        axes[i].imshow(scan[i].numpy()) # cmap can be changed to gray
        axes[i].set_title(f'Slice {i + 1}')
        axes[i].axis('off')

    plt.show()

def import_oct(path):
    # Load the e2e file as an EyeVolume object
    volume = ep.import_heyex_e2e(path)
    volume_data = torch.tensor(volume.data, dtype=torch.float32)
    return volume_data

def interpolate_scan(scan, method='nearest'):
    scan = scan.numpy()
    depth, height, width = scan.shape

    current_depths = np.arange(depth)
    desired_depths = np.linspace(0, depth - 1, 2 * depth)

    interpolator = RegularGridInterpolator(
        (current_depths, np.arange(height), np.arange(width)),
        scan,
        method=method,  # or 'slinear'/'linear'(has very incorrect data on earlier slices) or 'cubic'(takes alot of computing)
        bounds_error=False,
        fill_value=None  # fills out of bound values with the edge values
    )

    depth_grid, height_grid, width_grid = np.meshgrid(desired_depths, np.arange(height), np.arange(width),
                                                      indexing='ij')

    return  torch.tensor(interpolator((depth_grid, height_grid, width_grid)), dtype=torch.float32)
