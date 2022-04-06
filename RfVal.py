from sklearn.ensemble import RandomForestRegressor
import time
import pickle
import numpy as np
import open3d as o3d
import sklearn.metrics 

pickle_file = "C:\\data_for_learning\\full_shape_val_data.pkl" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
pickle_file = "C:\\data_for_learning\\Regression.pickle" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)

objectlist=[]
for i in range(len(data)):
    if (data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle' or data[i]['semantic class'] == 'Bowl' or data[i]['semantic class'] == 'Mug'):
        objectlist.append(data[i])

MSE=[]
MSPE=[]
R2=[]
Var=[]
for i in range(len(objectlist)):
        xyz=np.asarray(objectlist[i]['full_shape']['coordinate'])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector((xyz))
        Aff_v1=objectlist[i]['full_shape']['label']['grasp']
        Aff_v2=objectlist[i]['full_shape']['label']['wrap_grasp']
        Aff_v=np.maximum(Aff_v1,Aff_v2)

        np_colors=np.zeros((len(pcd.points),2))
        np_colors=(np.concatenate((Aff_v,np_colors),axis=1))
        pcd.colors=o3d.utility.Vector3dVector(np_colors)
        pcd=pcd.voxel_down_sample(voxel_size=0.02)
        Aff_v=np.asarray(np.asarray(pcd.colors)[:, :1])
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=5))
        pcd.paint_uniform_color([0, 0, 0])

        fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=10))
        fph = np.array(np.asarray(fph.data)).T
        
        aff=reg.predict(fph)
        

        aff = np.reshape(aff,(len(aff),1))
        mSE=sklearn.metrics.mean_squared_error(Aff_v,aff)
        mSPE=sklearn.metrics.mean_absolute_percentage_error(Aff_v,aff)
        r2=sklearn.metrics.r2_score(Aff_v,aff)
        var=sklearn.metrics.explained_variance_score(Aff_v,aff)
        Var.append(var)
        MSE.append(mSE)
        MSPE.append(mSPE)
        R2.append(r2)



        np_colors=np.zeros((len(pcd.points),2))
        np_colors=(np.concatenate((aff,np_colors),axis=1))
        #np_colors=(np.concatenate((aff*2,aff*2,aff*2),axis=1))
        pcd.colors=o3d.utility.Vector3dVector(np_colors)
        #o3d.visualization.draw_geometries([pcd])


print("Mean squared error",np.asarray(MSE).mean())
print("Mean absolute percentage error",np.asarray(MSPE).mean())
print("R² score, the coefficient of determination",np.asarray(R2).mean())
print("explained_variance_score",np.asarray(Var).mean())

