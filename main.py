from utils import normalize, show_scan, import_oct, interpolate_scan
import torch
import numpy as np

file_path = "eye.e2e"
scan = import_oct(file_path)
print(scan.shape)

scan = normalize(scan)
#print(scan)
#print("single slice:", scan[0][0])
#print(scan[0][0].shape)

#show_scan(scan)

#scan = interpolate_scan(scan)
scan = interpolate_scan(scan, method='linear')

print("Original shape:", scan.shape)
print("Interpolated shape:", scan.shape)

show_scan(scan, 5, 10)