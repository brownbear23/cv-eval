import numpy as np

npy_file_path = "../media/frames/chair_deg0_light1/chair_deg0_light1_48_depth.npy"

depth_array = np.load(npy_file_path)
depth_value = round(float(depth_array[647, 882]), 2) #y, x

print(depth_value)
