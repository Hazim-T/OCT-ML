import cv2
import matplotlib.pyplot as plt
import eyepy as ep
import skimage
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import center_of_mass, minimum_filter


def import_oct(path):
    volume = ep.import_heyex_e2e(path)
    volume_data = torch.tensor(volume.data, dtype=torch.float32)
    return volume_data


def normalize(data):
    return data / 255.0


def show_slices(scan, x=5, y=5, title='plot'):
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


def align_oct(oct_tensor, reference_slice_index=12, threshold=0, special_slice=None):
    # special slice is written in number not index
    if special_slice is not None:
        special_slice -= 1

    num_slices, height, width = oct_tensor.shape
    aligned_slices = []

    reference_slice = np.array(oct_tensor[reference_slice_index])
    reference_mask = reference_slice > threshold
    reference_com = center_of_mass(reference_mask)

    # Handle the rest of the slices first, excluding the special slice
    for i in range(num_slices):
        if i == special_slice:
            continue
        slice_ = np.array(oct_tensor[i])

        mask = slice_ > threshold
        com = center_of_mass(mask)

        shift_x = int(reference_com[0] - com[0])
        shift_y = int(reference_com[1] - com[1])

        aligned_slice = np.roll(slice_, shift=(shift_x, shift_y), axis=(0, 1))
        aligned_slices.append(aligned_slice)

    if special_slice is not None:
        slice_ = np.array(oct_tensor[special_slice])
        com_y, com_x = center_of_mass(slice_)
        center_y = height // 2
        shift_y = int(center_y - com_y)

        shifted_slice = np.zeros_like(slice_)
        if shift_y > 0:
            shifted_slice[shift_y:, :] = slice_[:height - shift_y, :]
        elif shift_y < 0:
            shifted_slice[:height + shift_y, :] = slice_[-shift_y:, :]
        else:
            shifted_slice = slice_

        aligned_slices.insert(special_slice, shifted_slice)

    aligned_tensor = np.stack(aligned_slices)

    return torch.tensor(aligned_tensor, dtype=torch.float32)


def segment_eye_components(tensor):
    tensor = np.array(tensor)
    segmented_tensor = np.zeros_like(tensor)

    for i in range(tensor.shape[0]):
        slice_img = tensor[i]
        smoothed_slice = minimum_filter(slice_img, size=(2, 2))

        _, binary_img = cv2.threshold(smoothed_slice, np.mean(slice_img) * 0.5, 255, cv2.THRESH_BINARY)

        labels = skimage.measure.label(binary_img, connectivity=2)

        props = skimage.measure.regionprops(labels)
        if props:
            largest_component = max(props, key=lambda x: x.area)
            mask = labels == largest_component.label
            segmented_tensor[i][mask] = tensor[i][mask]

    segmented_tensor = align_oct(segmented_tensor)
    return segmented_tensor


def plot_oct_3d(tensor):
    mask = tensor > 0.0
    x, y, z = np.indices(tensor.shape)

    x_nonzero = x[mask]
    y_nonzero = y[mask]
    z_nonzero = z[mask]
    values_nonzero = tensor[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_nonzero, y_nonzero, z_nonzero, c=values_nonzero, cmap='gray', s=2)
    ax.set_xlim(0, tensor.shape[0])
    ax.set_ylim(0, tensor.shape[1])
    ax.set_zlim(0, tensor.shape[2])
    ax.view_init(roll=100, azim=195)
    ax.set_axis_off()
    plt.show()