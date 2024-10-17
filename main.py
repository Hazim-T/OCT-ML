from utils import *

# Shape: [25, 496, 512] - 25 slices of 496x512
file_path = "eye.e2e"
scan = import_oct(file_path)
print("Original shape:", scan.shape)
show_slices(scan, title="Original")

scan = align_oct(scan, special_slice=1)
print(scan.shape)
show_slices(scan, title="Aligned")

scan = interpolate_scan(scan, method='linear')
print("Interpolated shape:", scan.shape)
show_slices(scan, x=5, y=10, title="Aligned + Interpolated")

scan = normalize(scan)
scan = segment_eye_components(scan)
plot_oct_3d(scan)