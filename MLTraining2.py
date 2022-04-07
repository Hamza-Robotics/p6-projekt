from cProfile import label
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
    if (data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle' or data[i]['semantic class'] == 'Mug'):
        objectlist.append(data[i])
    

x=[]
y=[]
labels=np.asarray([])
for i in range(len(objectlist)):
    xyz=np.asarray(objectlist[i]['full_shape']['coordinate'])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((xyz))


    Aff_v1=objectlist[i]['full_shape']['label']['grasp']
    Aff_v2=objectlist[i]['full_shape']['label']['wrap_grasp']
    np_colors=np.zeros((len(pcd.points),1))
    np_colors=(np.concatenate((Aff_v1,Aff_v2,np_colors),axis=1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((xyz))
    pcd.colors=o3d.utility.Vector3dVector(np_colors)

    pcd=pcd.voxel_down_sample(voxel_size=0.02)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=30))
    fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=55))

    fph = np.array(np.asarray(fph.data)).T
    #x.append(fph)
    for k in range(len(np.array(pcd.points))):
        x.append(fph[k])
    #labels=[objectlist[i]['full_shape']['label']['grasp'], objectlist[i]['full_shape']['label']['wrap_grasp'], objectlist[i]['full_shape']['label']['contain']]
    #labels = np.reshape(labels,(np.shape(labels)[0],np.shape(labels)[1]))
    if i==0:
        y=[np.asarray(np.asarray(pcd.colors)[:,:1]),np.asarray(np.asarray(pcd.colors)[:,1:2])]

    else:    
        y=np.append(y,([np.asarray(np.asarray(pcd.colors)[:,:1]),np.asarray(np.asarray(pcd.colors)[:,1:2])]),axis=1)

    #print(np.shape([np.asarray(np.asarray(pcd.colors)[:,:1]),np.asarray(np.asarray(pcd.colors)[:,1:2])]))

x=np.asarray(x)
y=np.asarray(y).reshape(np.shape(y)[1],np.shape(y)[0])
print(np.shape(y))
print(np.shape(x))

np.save('C:\\data_for_learning\\x_values.npy', x)
np.save('C:\\data_for_learning\\y_values.npy', y)
np.save
