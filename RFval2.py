from sklearn.ensemble import RandomForestRegressor
import time
import pickle
import numpy as np
import open3d as o3d
import sklearn.metrics 
def CenterOfPCD(PCD):
    m_xyz=np.mean(PCD,axis=0)
    L=np.linalg.norm(m_xyz-PCD,axis=1).reshape((len(PCD),1))
    L=(L-min(L))/(max(L)-min(L))
    return L

print("pickle val data")
pickle_file = "C:\\data_for_learning\\full_shape_val_data.pkl" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
print("pickle learner")
pickle_file = "C:\\data_for_learning\\RegressionNEW2.pickle" ### Write path for the full_shape_val_data.pkl file ###
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
MAE=[]
Time=[]
for i in range(10):
    if True:
        xyz=np.asarray(objectlist[i]['full_shape']['coordinate'])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector((xyz))

        Aff_v1=objectlist[i]['full_shape']['label']['grasp']
        Aff_v2=objectlist[i]['full_shape']['label']['wrap_grasp']
        np_colors=np.zeros((len(pcd.points),1))
        np_colors=(np.concatenate((Aff_v1,Aff_v2,np_colors),axis=1))
        pcd.colors=o3d.utility.Vector3dVector(np_colors)
        pcd=pcd.voxel_down_sample(voxel_size=0.02)
        Aff_g=np.asarray(np.asarray(pcd.colors)[:, :1]).copy()
        Aff_W=np.asarray(np.asarray(pcd.colors)[:, 1:2]).copy()


        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=10))
        pcd.paint_uniform_color([0, 0, 0])


        fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=50))
        L=CenterOfPCD(np.asarray(pcd.points))
        fph = np.array(np.asarray(fph.data)).T
        #print("L:",np.shape(L))
        #print("fph:",np.shape(fph))

        fph=np.append(fph,L,axis=1)

        #print("fph:",np.shape(fph))
        begin=time.time()

        aff=reg.predict(fph)
        timeittook=time.time()-begin
        #print("aff:",np.shape(aff))
        aff=np.asarray(aff).reshape((len(pcd.points),1))
        mSE=sklearn.metrics.mean_squared_error(Aff_g,aff)
        r2=sklearn.metrics.r2_score(Aff_g,aff)
        var=sklearn.metrics.explained_variance_score(Aff_g,aff)
        mAE=sklearn.metrics.mean_absolute_error(Aff_g, aff)


        np_colors=np.zeros((len(pcd.points),2))
        np_colors=(np.concatenate((aff,np_colors),axis=1))
        pcd.colors=o3d.utility.Vector3dVector(np_colors)
        #o3d.visualization.draw_geometries([pcd])
        #print(np.asarray(pcd.colors)==Aff_g)



        Var.append(var)
        MSE.append(mSE)
        R2.append(r2)
        MAE.append(mAE)
        Time.append(timeittook)



    #np_colors=np.zeros((len(pcd.points),2))
    #np_colors=(np.concatenate((aff,np_colors),axis=1))
    #np_colors=(np.concatenate((aff*2,aff*2,aff*2),axis=1))
    #pcd.colors=o3d.utility.Vector3dVector(np_colors*2)


print("Mean squared error",np.asarray(MSE).mean())
print("R² score, the coefficient of determination",np.asarray(R2).mean())
print("explained_variance_score",np.asarray(Var).mean())
print("Mean absolute error",np.asarray(MAE).mean())
print("average time to compute grasp", np.asarray(Time).mean())
print("  ss ")
print(" ss  ")

index=np.argmax(R2)
print("Lowest squared error",(np.asarray(MSE[index])))
print("Best R² score, the coefficient of determination",(np.asarray(R2[index])))
print("Best explained_variance_score",(np.asarray(Var[index])))
print("Best absolute error",(np.asarray(MAE[index])))
print("Best average time to compute grasp", (np.asarray(Time[index])))