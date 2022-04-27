import pickle
import open3d as o3d
import numpy as np


pickle_file = 'C:\\data_for_learning\\full_shape_train_data.pkl' ### Write path for the full_shape_val_data.pkl file ###

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
j = 0
#print(data[0]['full_shape']['label'])#['coordinate'])
objectlist=[]
while(1):
    pass
    break
for i in range(len(data)):
    if (data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle' or data[i]['semantic class'] == 'Bowl' or data[i]['semantic class'] == 'Mug'):
        for l in range(len(data[i])):
            data[i]['full_shape']['label']['grasp'][l] = data[i]['full_shape']['label']['grasp'][l][0]
            data[i]['full_shape']['label']['wrap_grasp'][l] = data[i]['full_shape']['label']['grasp'][l][0]
            data[i]['full_shape']['label']['contain'][l] = data[i]['full_shape']['label']['grasp'][l][0]

        objectlist.append(data[i])
        xyz=np.asarray(objectlist[j]['full_shape']['coordinate'])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector((xyz))
        pcd=pcd.voxel_down_sample(voxel_size=0.02)
        #o3d.visualization.draw_geometries([pcd])
        j+=1

x=[]
y=[]
for i in range(len(objectlist)):
    xyz=np.asarray(objectlist[i]['full_shape']['coordinate'])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((xyz))
    #o3d.visualization.draw_geometries([pcd])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=100))
    fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=200))
    fph = np.array(np.asarray(fph.data)).T
    #x.append(fph)
    for k in range(len(np.array(pcd.points))):
        x.append(fph[k])
    #labels=[objectlist[i]['full_shape']['label']['grasp'], objectlist[i]['full_shape']['label']['wrap_grasp'], objectlist[i]['full_shape']['label']['contain']]
    #labels = np.reshape(labels,(np.shape(labels)[0],np.shape(labels)[1]))
    labels = []
    for j in range(len(objectlist[i]['full_shape']['label']['grasp'])):
        labels.append(max(objectlist[i]['full_shape']['label']['grasp'][j],objectlist[i]['full_shape']['label']['wrap_grasp'][j]))
        #print(objectlist[i]['full_shape']['label']['grasp'][j], objectlist[i]['full_shape']['label']['wrap_grasp'][j], labels[j])
    #labels=np.asarray(labels).T
    #labels = np.reshape(labels,(np.shape(labels)[0]))
    for k in range(len(np.array(pcd.points))):
        y.append(labels[k])


x=np.asarray(x)
y=np.asarray(y)
print(np.shape(y))
print(np.shape(x))

np.save('C:\\data_for_learning\\x_values.npy', x)
np.save('C:\\data_for_learning\\y_values.npy', y)