import matplotlib.pyplot as plt
import eyepy as ep
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import center_of_mass


def import_oct(path):
    # Load the e2e file as an EyeVolume object
    volume = ep.import_heyex_e2e(path)
    volume_data = torch.tensor(volume.data, dtype=torch.float32)
    return volume_data


def normalize(data):
    #return data / 255.0
    return (data - data.min()) / (data.max() - data.min())


def show_scan(scan, x=5, y=5, title='plot'):
    fig, axes = plt.subplots(x, y, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(scan.shape[0]):
        axes[i].imshow(scan[i], cmap='gray')
        axes[i].set_title(f'Slice {i + 1}')
        axes[i].axis('off')

    plt.suptitle(title)
    plt.show()


def interpolate_scan(scan, method='linear'):
    scan = scan.numpy()
    depth, height, width = scan.shape

    current_depths = np.arange(depth)
    desired_depths = np.linspace(0, depth - 1, 2 * depth)

    interpolator = RegularGridInterpolator(
        (current_depths, np.arange(height), np.arange(width)),
        scan,
        method=method,  # method can be 'slinear'/'linear'(has very incorrect data on earlier slices) or 'cubic'(takes alot of computing)
        bounds_error=False,
        fill_value=None
    )

    depth_grid, height_grid, width_grid = np.meshgrid(desired_depths, np.arange(height), np.arange(width),
                                                      indexing='ij')

    return  torch.tensor(interpolator((depth_grid, height_grid, width_grid)), dtype=torch.float32)


def align_oct(oct_tensor, reference_slice_index=12, threshold=0):
    num_slices, height, width = oct_tensor.shape
    aligned_slices = []

    reference_slice = oct_tensor[reference_slice_index].numpy()
    reference_mask = reference_slice > threshold
    reference_com = center_of_mass(reference_mask)

    # Special handling for slice 1 as it is always shifted higher than the rest of the slices
    slice_1 = oct_tensor[0].numpy()
    com_y_1, com_x_1 = center_of_mass(slice_1)
    center_y = height // 2
    shift_y_1 = int(center_y - com_y_1)

    shifted_slice_1 = np.zeros_like(slice_1)
    if shift_y_1 > 0:
        shifted_slice_1[shift_y_1:, :] = slice_1[:height - shift_y_1, :]
    elif shift_y_1 < 0:
        shifted_slice_1[:height + shift_y_1, :] = slice_1[-shift_y_1:, :]
    else:
        shifted_slice_1 = slice_1

    aligned_slices.append(shifted_slice_1)

    # Rest of the slices, so we start from 2nd slice
    for i in range(1, oct_tensor.shape[0]):
        slice_ = oct_tensor[i].numpy()

        mask = slice_ > threshold
        com = center_of_mass(mask)

        shift_x = int(reference_com[0] - com[0])
        shift_y = int(reference_com[1] - com[1])

        aligned_slice = np.roll(slice_, shift=(shift_x, shift_y), axis=(0, 1))
        aligned_slices.append(aligned_slice)

    aligned_tensor = np.stack(aligned_slices)

    return torch.tensor(aligned_tensor, dtype=torch.float32)