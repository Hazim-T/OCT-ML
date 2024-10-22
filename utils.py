import cv2
import matplotlib.pyplot as plt
import eyepy as ep
import skimage
import torch
import numpy as np
from scipy.ndimage import center_of_mass, minimum_filter
import torch.nn.functional as F
import os


def load_eye_tensors(full_path):
    with ep.io.HeE2eReader(full_path) as reader:
        data = reader.volumes

    eye_tensors = []
    for eye_volume in data:
        bscans = eye_volume[0:]
        bscan_list = [bscan.data for bscan in bscans]
        eye_tensor = np.stack(bscan_list)

        eye_tensors.append(eye_tensor)

    return np.array(eye_tensors)


def load_eye_tensors_with_labels(data_folder, df):
    rejected = ["nan", "-", "false", "treated active", "treated inactive", "myopic cnv"]
    wet_amd = ["mac. neovas. type 1-occult", "mac. neovas. type 2-classic", "mac. neovas. mixed type",
               "mac. neovas. type 1-occult-inactive", "end stage exudutive amd"]
    dry_amd = ["non-exudutive amd low risk", "non-exudutive amd high risk", "vitelliform deg", "geographic atrophy"]

    wet, dry = 0, 0
    accepted = []

    for i in range(len(df)):
        if (str(df[i][1]).lower() not in rejected or str(df[i][2]).lower() not in rejected) and (
                str(df[i][0]).lower() not in rejected):
            accepted.append(df[i])

            if str(df[i][2]).lower() in wet_amd or str(df[i][1]).lower() in wet_amd:
                wet += 1
            elif str(df[i][2]).lower() in dry_amd or str(df[i][1]).lower() in dry_amd:
                dry += 1
            else:
                print("problematic: ", df[i])

    accepted = np.array(accepted)
    eye_data = []
    corrupted_counter = 0
    for scan_info in accepted:
        filename, first_eye_label, second_eye_label = scan_info[0], str(scan_info[1]).lower(), str(scan_info[2]).lower()
        full_path = os.path.join(data_folder, filename)

        try:
            if os.path.exists(full_path):
                eye_tensors = load_eye_tensors(full_path)
                print(f"Loaded {filename}")
                if first_eye_label not in rejected:
                    eye_data.append((eye_tensors[0], 1 if first_eye_label in wet_amd else 0))
                if second_eye_label not in rejected:
                    eye_data.append((eye_tensors[1], 1 if second_eye_label in wet_amd else 0))
            else:
                print(f"File {filename} not found in {data_folder}")
        except:
            try:
                if first_eye_label in rejected:
                    eye_data.append((eye_tensors[0], 1 if second_eye_label in wet_amd else 0))
            except Exception as e:
                print(f"File {filename} issue", e)
                corrupted_counter += 1

    print(f"Bad file count: {corrupted_counter}",)
    return np.array(eye_data, dtype=object)


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


def resize_slices_2d(tensor_slices):
    original_width = tensor_slices.shape[2]
    original_height = tensor_slices.shape[1]

    if original_width != 512 or original_height != 496:
        resized_slices = F.interpolate(tensor_slices.unsqueeze(0), size=(496, 512), mode='bilinear', align_corners=False)
        return resized_slices.squeeze(0)
    return tensor_slices


def align_oct(oct_tensor, reference_slice_index=17, threshold=0, special_slice=None):
    if special_slice is not None:
        special_slice -= 1

    num_slices, height, width = oct_tensor.shape
    aligned_slices = []

    reference_slice = np.array(oct_tensor[reference_slice_index])
    reference_mask = reference_slice > threshold
    reference_com = center_of_mass(reference_mask)

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


def filter_and_normalize(tuple_tensor):
    labels = []
    accepted_eyes = []
    eyes = tuple_tensor[:, 0]

    for i in range(len(eyes)):
        if eyes[i].shape[0] == 25:
            accepted_eyes.append(align_oct(normalize(eyes[i])))
            labels.append(tuple_tensor[:, 1][i])

    return np.array(accepted_eyes), np.array(labels)


def segment_eye_components(tensor):
    tensor = np.array(tensor)
    segmented_tensor = np.zeros_like(tensor)

    for i in range(tensor.shape[0]):
        slice_img = tensor[i]
        smoothed_slice = minimum_filter(slice_img, size=(2, 2))

        _, binary_img = cv2.threshold(smoothed_slice, np.mean(slice_img) * 0.6, 255, cv2.THRESH_BINARY)

        labels = skimage.measure.label(binary_img, connectivity=2)

        props = skimage.measure.regionprops(labels)
        if props:
            largest_component = max(props, key=lambda x: x.area)
            mask = labels == largest_component.label
            segmented_tensor[i][mask] = tensor[i][mask]

    segmented_tensor = align_oct(segmented_tensor)
    return segmented_tensor


def interpolate_scan(scan, max_depth=25):
    scan = scan.unsqueeze(0).unsqueeze(0)
    scan = F.interpolate(scan, size=(max_depth, 496, 512), mode='trilinear', align_corners=False)
    scan = scan.squeeze(0).squeeze(0)
    return scan


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