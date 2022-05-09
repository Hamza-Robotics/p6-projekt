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
pickle_file = "C:\\data_for_learning\\RegressionGrasp.pickle" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)
reg.verbose=False
print(reg.feature_importances_)

print(reg.get_params())
objectlist=[]
for i in range(len(data)):
    if (data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle' or data[i]['semantic class'] == 'Bowl' or data[i]['semantic class'] == 'Mug'):
        objectlist.append(data[i])

MSE=[]
MSPE=[]
R2=[]
Var=[]
MAE=[]
PE=[]
Time=[]
for i in range(len(objectlist)):
    if True:
        xyz=np.asarray(objectlist[i]['full_shape']['coordinate'])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector((xyz))
 

        Aff_v1=objectlist[i]['full_shape']['label']['grasp']
        Aff_v2=objectlist[i]['full_shape']['label']['wrap_grasp']
        np_colors=np.zeros((len(pcd.points),1))
        np_colors=(np.concatenate((Aff_v1,Aff_v2,np_colors),axis=1))
        pcd.colors=o3d.utility.Vector3dVector(np_colors)
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        radius = diameter * 100
        camera = [1,0,diameter]
        _, pt_map = pcd.hidden_point_removal(camera, radius)
        pcd = pcd.select_by_index(pt_map)
        
        #pcd=pcd.voxel_down_sample(voxel_size=0)
        Aff_g=np.asarray(np.asarray(pcd.colors)[:, :1]).copy()
        Aff_W=np.asarray(np.asarray(pcd.colors)[:, 1:2]).copy()

        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*3, max_nn=100))
        pcd.paint_uniform_color([0, 0, 0])

        fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*11, max_nn=300))
        L=CenterOfPCD(np.asarray(pcd.points))
        fph = np.array(np.asarray(fph.data)).T
        #print("L:",np.shape(L))
        #print("fph:",np.shape(fph))

        fph=np.append(fph,L,axis=1)

        print("fph:",np.shape(fph))
        begin=time.time()
        print("pints:",np.shape(np.asarray(pcd.points)))
        print("len:  ", len(pcd.points), "another",np.shape(pcd.points)[0])
        aff=reg.predict(fph)
        timeittook=time.time()-begin
        #print("aff:",np.shape(aff))
        aff=np.asarray(aff).reshape((len(pcd.points),1))
        mSE=sklearn.metrics.mean_squared_error(Aff_g,aff,squared=False)
        r2=sklearn.metrics.r2_score(Aff_g,aff)
        var=sklearn.metrics.explained_variance_score(Aff_g,aff)
        mAE=sklearn.metrics.mean_absolute_error(Aff_g, aff)
        pe=sklearn.metrics.mean_absolute_percentage_error(Aff_g,aff)

        maxaff=np.argmax(aff)


        np_colors=np.zeros((len(pcd.points),2))
        np_colors=(np.concatenate((aff,np_colors),axis=1))
        np_colors[maxaff][:]=0
        np_colors[maxaff][2]=1     
        pcd.colors=o3d.utility.Vector3dVector(np_colors)

        print(np.shape(aff))
        #pcd.colors=o3d.utility.Vector3dVector(np.concatenate((aff,np.asarray(np.asarray(pcd.colors)[:, :1]),np.asarray(np.asarray(pcd.colors)[:, 1:2])),axis=1))

        o3d.visualization.draw_geometries([pcd])
        #print(np.asarray(pcd.colors)==Aff_g)



        Var.append(var)
        MSE.append(mSE)
        R2.append(r2)
        MAE.append(mAE)
        Time.append(timeittook)
        PE.append(pe)



    #np_colors=np.zeros((len(pcd.points),2))
    #np_colors=(np.concatenate((aff,np_colors),axis=1))
    #np_colors=(np.concatenate((aff*2,aff*2,aff*2),axis=1))
    #pcd.colors=o3d.utility.Vector3dVector(np_colors*2)


print("Root-mean-square deviation",np.asarray(MSE).mean())
print("R² score, the coefficient of determination",np.asarray(R2).mean())
print("explained_variance_score",np.asarray(Var).mean())
print("Mean absolute error",np.asarray(MAE).mean())
print("average time to compute grasp", np.asarray(Time).mean())
print("average Mean absolute percentage error", np.asarray(PE).mean())



