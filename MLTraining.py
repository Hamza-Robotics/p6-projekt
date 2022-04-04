import pickle
import open3d as o3d
import numpy as np


pickle_file = "C:\\full-shape\\full_shape_train_data.pkl" ### Write path for the full_shape_val_data.pkl file ###

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
y = 0
print(data[0]['full_shape']['coordinate'])

for i in range(10):

    xyz=np.asarray(data[i]['full_shape']['coordinate'])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((xyz))

    o3d.visualization.draw_geometries([pcd])
