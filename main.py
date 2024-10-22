from utils import *

# Shape: (2, [25, 496, 512]) - 2 eyes, each have 25 slices of 496x512
file_path = "eye.e2e"
scan = load_eye_tensors(file_path)
left_eye = scan[0]
right_eye = scan[1]

print("Original shape:", left_eye.shape)
show_slices(left_eye, title="Original")

left_eye = align_oct(left_eye, special_slice=1)
print(left_eye.shape)
show_slices(left_eye, title="Aligned")

left_eye = interpolate_scan(left_eye)
print("Interpolated shape:", left_eye.shape)
show_slices(left_eye, title="Aligned + Interpolated")

left_eye = normalize(left_eye)
left_eye = segment_eye_components(left_eye)
plot_oct_3d(left_eye)