from turtle import color, position
import cv2
from sklearn.ensemble import RandomForestRegressor
import time
import pickle
import numpy as np
import open3d as o3d
import copy
import sklearn.metrics
import random



print("Pickle learner")
pickle_file = "C:\\data_for_learning\\RegressionGrasp2.pickle" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)        
print("pickle Learned")
#reg.verbose=False

xyz=np.load('RealData\\'+str(1)+'.npy')

def Rotation(rot,iteration,resulution):
    Circle=np.linspace(0, 2*np.pi, resulution)
    
    angle=rot-Circle[iteration]

    if angle>2*np.pi:  
        angle=angle-2*np.pi
        return angle
    if angle<0:
        angle=angle+2*np.pi
        return angle
    else:
        return angle

def CenterOfPCD(PCD):
    m_xyz=np.mean(PCD,axis=0)
    L=np.linalg.norm(m_xyz-PCD,axis=1).reshape((len(PCD),1))
    L=(L-min(L))/(max(L)-min(L))
    return L

def grasp_Positions(pcd,aff):

    pcd=ColorAffordance(aff,pcd,[0,0,1])
    k=80
    position=np.expand_dims(np.squeeze(np.asarray(aff[aff.argsort(axis=0)[-k:]])),axis=1)      
    #np.asarray(pcd.colors)[aff.argsort(axis=0)[-k:], :] = [0, 1, 1]
    #o3d.visualization.draw_geometries([pcd])
    try:
        pass

          
    except:
        print("no affordance")
        return np.asarray([0]),np.asarray([0]),np.asarray([0])
    PfLEO_OBB = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T = np.eye(4)
    T[:3, :3]= PfLEO_OBB.R
    poses=[]
    X=np.asarray([])

    

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    pcd_ = pcd.select_by_index(position)
    pcd_.paint_uniform_color([0,0,0])
    for i in range(len(pcd_.points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d((pcd_.points)[i], 0.007*factor)
        pcd2 = pcd.select_by_index(idx)

        #np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        #o3d.visualization.draw_geometries([pcd])

        cov_m=np.cov(np.asarray(pcd2.points).transpose())
        cov_m[np.isnan(cov_m)] = 0
        S=np.linalg.svd(cov_m,False,False)  #no full svd nor UV
        k1k2=([S[0],S[1]])
        m_curv=([(S[0]+S[1]/2)])
        g_curv=([(S[0]*S[1]/2)])
        
        for r in range(8):
            for p in range(8):
                for y in range(8):
                    T[0, 3]=np.asarray(pcd_.points)[i][0]
                    T[1, 3]=np.asarray(pcd_.points)[i][1]
                    T[2, 3]=np.asanyarray(pcd_.points)[i][2]
                    R,_=cv2.Rodrigues(PfLEO_OBB.R)
                    Rot_xyz=(np.array([[Rotation(R[0],r,10),Rotation(R[1],p,10),Rotation(R[2],y,10)]]))
                    T[:3, :3],_=cv2.Rodrigues(Rot_xyz)
                    print(T[:3, :3])
                    rpy,_=cv2.Rodrigues(T[:3, :3])

                    
                    x=np.concatenate(([T[0, 3]/np.median(pcd_.points[0],axis=0),T[1, 3]/np.median(pcd_.points[0],axis=0),T[2, 3]/np.median(pcd_.points[0],axis=0)],R[0]-rpy[0],R[1]-rpy[1],R[2]-rpy[2],(pcd_.colors)[i],
                    k1k2,m_curv,g_curv))
                    poses.append(T)
                    if  r==0 and p==0 and y==0 and i==0:
                        X=np.transpose(np.expand_dims(x,axis=1))
                    else:
                        X=np.concatenate((X,np.transpose(np.expand_dims(x,axis=1))),axis=0)

            #X=np.concatenate( (np.concatenate((X,np.transpose(np.expand_dims(x,axis=1))),axis=1),x_e),axis=0)


    return np.asarray(poses), np.asarray(X), pcd_

def ColorAffordance(aff,pcd,color):
    np_colors=np.zeros((len(pcd.points),2))
    np_colors=(np.concatenate((aff,np_colors),axis=1))
    #maxaff=np.argmax(aff)
    #np_colors[maxaff][:]=[0,0,1]

    pcd.colors=o3d.utility.Vector3dVector(np_colors)

    return pcd
factor=3.8
def SVD_Principal_Curvature(Pointcloud,radius):
    k1k2=[]
    m_curv=[]
    g_curv=[]
    pcd_tree = o3d.geometry.KDTreeFlann(Pointcloud)

    for i in range(len(Pointcloud.points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d((Pointcloud.points)[i], radius*factor)
        pcd_ = pcd.select_by_index(idx)
        Pointcloud.paint_uniform_color([0,0,0])

        np.asarray(Pointcloud.colors)[idx[1:], :] = [0, 0, 1]
        #o3d.visualization.draw_geometries([Pointcloud])
        cov_m=np.cov(np.asarray(pcd_.points).transpose())
        cov_m[np.isnan(cov_m)] = 0
        S=np.linalg.svd(cov_m,False,False)  #no full svd nor UV
        k1k2.append([S[0],S[1]])
        m_curv.append([(S[0]+S[1]/2)])
        g_curv.append([(S[0]*S[1]/2)])

    return np.concatenate((np.asarray(k1k2), np.asarray(m_curv), np.asarray(g_curv)),axis=1)

def Extract_Feature(pcd):
    #Non Normalized distance: 
    L=CenterOfPCD(np.asarray(pcd.points))
#curvature features:
    cur=SVD_Principal_Curvature(pcd,0.007)
    x=np.append(cur,L,axis=1)
# Fast Feature
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.1*factor))
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamRadius(radius=0.06*factor))
    
    fph = np.array(np.asarray(fph.data)).T
    fph=np.append(fph,L,axis=1)
    
    x=np.append(cur,fph,axis=1)

    return x

for i in range(7):
    if True:
        xyz=np.load('RealData\\'+str(i+1)+'.npy')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector((xyz))
        pcd.paint_uniform_color([0,0,0])
        pcd=pcd.voxel_down_sample(voxel_size=0.007)
        x=Extract_Feature(pcd)
        aff=reg.predict(x)
        aff=np.asarray(aff).reshape((len(pcd.points),1))

        print("x=[")
        T,X,pcd_=grasp_Positions(pcd,aff)
        for j in range(len(T)):
            T_=T[j]
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh_t = copy.deepcopy(mesh).transform(T_)
            mesh_t.scale(0.1, center=mesh_t.get_center())

            idx=np.argwhere((np.asarray(pcd.points) == np.asarray([[T_[0, 3],T_[1, 3],T_[2, 3]]])).all(axis=1))
            np.asarray(pcd.colors)[idx, :] = [0, 0, 1]

            o3d.visualization.draw_geometries([pcd,mesh_t])

            print((T[j]==T[j+20]),",")

            print(X[j]==X[j+20])
 





