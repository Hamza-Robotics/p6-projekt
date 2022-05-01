from sklearn.ensemble import RandomForestRegressor
import time
import pickle
import numpy as np
import open3d as o3d
import copy
import sklearn.metrics
xyz=np.load('RealData\\'+str(1)+'.npy')

def CenterOfPCD(PCD):
    m_xyz=np.mean(PCD,axis=0)
    L=np.linalg.norm(m_xyz-PCD,axis=1).reshape((len(PCD),1))
    L=(L-min(L))/(max(L)-min(L))
    return L


print("pickle learner")
pickle_file = "C:\\data_for_learning\\RegressionGrasp.pickle" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)

reg.verbose=False


MSE=[]
MSPE=[]
R2=[]
Var=[]
MAE=[]
PE=[]
Time=[]
def grasp_Position(pcd, position):
    PfLEO_OBB = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T = np.eye(4)
    T[:3, :3]= PfLEO_OBB.R
    aff=pcd.select_by_index([aff_m])
    aff.paint_uniform_color([1, 1, 0])

    T[0, 3]=np.asanyarray(aff.points)[0,0]
    T[1, 3]=np.asanyarray(aff.points)[0,1]
    T[2, 3]=np.asanyarray(aff.points)[0,2]
    print(np.asanyarray(aff.points),"and         t=",(T))
    mesh_t = copy.deepcopy(mesh).transform(T)
    mesh_t.scale(0.1, center=mesh_t.get_center())

    return T,mesh_t

for i in range(7):
    if True:
        xyz=np.load('RealData\\'+str(i+1)+'.npy')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector((xyz))
        
        pcd=pcd.voxel_down_sample(voxel_size=0.01)
        L=CenterOfPCD(np.asarray(pcd.points))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=30))
        pcd.paint_uniform_color([0, 0, 0])

        fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*5, max_nn=50))
        L=CenterOfPCD(np.asarray(pcd.points))
        fph = np.array(np.asarray(fph.data)).T
        #print("L:",np.shape(L))
        #print("fph:",np.shape(fph))
        fph=np.append(fph,L,axis=1)

        #fph=np.append(fph,L,axis=1)

        print("fph:",np.shape(fph))
        begin=time.time()
        print("pints:",np.shape(np.asarray(pcd.points)))
        print("len:  ", len(pcd.points), "another",np.shape(pcd.points)[0])
        aff=reg.predict(fph)
        timeittook=time.time()-begin
        print("aff:",np.shape(aff))
        aff=np.asarray(aff).reshape((len(pcd.points),1))
 
        np_colors=np.zeros((len(pcd.points),2))
        np_colors=(np.concatenate((aff,np_colors),axis=1))
        maxaff=np.argmax(aff)
        np_colors[maxaff][:]=0
        np_colors[maxaff][2]=1 
        pcd.colors=o3d.utility.Vector3dVector(np_colors)

        print(np.shape(aff))
        #pcd.colors=o3d.utility.Vector3dVector(np.concatenate((aff,np.asarray(np.asarray(pcd.colors)[:, :1]),np.asarray(np.asarray(pcd.colors)[:, 1:2])),axis=1))

        aff_m=np.argmax(np.asanyarray(pcd.colors)[:,0])
        
        T,mesh_t=grasp_Position(pcd,aff_m)



        o3d.visualization.draw_geometries([pcd,mesh_t])
        #print(np.asarray(pcd.colors)==Aff_g

    #np_colors=np.zeros((len(pcd.points),2))
    #np_colors=(np.concatenate((aff,np_colors),axis=1))
    #np_colors=(np.concatenate((aff*2,aff*2,aff*2),axis=1))
    #pcd.colors=o3d.utility.Vector3dVector(np_colors*2)




