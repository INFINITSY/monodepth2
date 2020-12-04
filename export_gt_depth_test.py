import os
import numpy as np
from kitti_utils import generate_depth_map

gt_depths = []

for i in range(107):

    calib_dir = "kitti_data"
    velo_filename = os.path.join("kitti_data/01",
                                 "velodyne_points/data", "{:010d}.bin".format(i))
    gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)

    gt_depths.append(gt_depth.astype(np.float32))

output_path = os.path.join("kitti_data/01", "gt_depths.npz")

np.savez_compressed(output_path, data=np.array(gt_depths))
