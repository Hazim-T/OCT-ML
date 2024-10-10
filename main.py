from utils import normalize, show_scan, import_oct, interpolate_scan, align_oct

# Shape: [25, 496, 512] - 25 slices of 496x512
file_path = "eye.e2e"
scan = import_oct(file_path)
print("Original shape:", scan.shape)
show_scan(scan, title="Original")

scan = align_oct(scan)
show_scan(scan, title="Aligned")

scan = interpolate_scan(scan, method='linear')
print("Interpolated shape:", scan.shape)
show_scan(scan, x=5, y=10, title="Aligned + Interpolated")

scan = normalize(scan)