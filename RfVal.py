from sklearn.ensemble import RandomForestRegressor
import time
import pickle
import numpy as np
import open3d as o3d

pickle_file = "C:\\full-shape\\full_shape_val_data.pkl" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
pickle_file = "C:\\full-shape\\Regression.pickle" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)

objectlist=[]
for i in range(len(data)):
    if (data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle' or data[i]['semantic class'] == 'Bowl' or data[i]['semantic class'] == 'Mug'):
        objectlist.append(data[i])

for i in range(10):
        xyz=np.asarray(objectlist[i]['full_shape']['coordinate'])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector((xyz))
        pcd=pcd.voxel_down_sample(voxel_size=0.02)
        pcd.paint_uniform_color([0, 0, 0])
        fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=10))
        fph = np.array(np.asarray(fph.data)).T

        aff=reg(fph)

        #np_colors=np.zeros((len(pcd.points),2))
        #np_colors=(np.concatenate((np_colors,objectlist[i]['full_shape']['label']['grasp']),axis=1))
        #pcd.colors=o3d.utility.Vector3dVector(np_colors)
        o3d.visualization.draw_geometries([pcd])
