import pickle 
import numpy as np
import open3d as o3d
import sklearn.metrics 
print("hello")
pickle_file = "C:\\data_for_learning\\partial_val_data.pkl" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
pickle_file = "C:\\data_for_learning\\DebugState.pickle"
    #RegressionRF.pickle" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)


objectlist=[]
for i in range(len(data)):
    if (data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle' or data[i]['semantic class'] == 'Bowl'):
        objectlist.append(data[i])
        #data[i]['semantic class'] == 'Mug'




def Rotation(rot,iteration,resulution):
    Circle=np.linspace(0, 2*np.pi, resulution)
    
    angle=Circle[0]
    
    #if angle>2*np.pi:  
      #  angle=angle-2*np.pi
     #   return angle
    #if angle<0:
     #   angle=angle+2*np.pi
    #    return angle
   # else:
    return angle

def CenterOfPCD(PCD):
    m_xyz=np.mean(PCD,axis=0)
    L=np.linalg.norm(m_xyz-PCD,axis=1).reshape((len(PCD),1))
    L=(L-min(L))/(max(L)-min(L))
    return L

def grasp_Positions(pcd,aff,factor):

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
    X=np.asarray([])
    Poses=np.asarray([])

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    pcd_ = pcd.select_by_index(position)
    pcd_.paint_uniform_color([0,0,0])
    #for i in range(len(pcd_.points)):
    for i in range(1):
        [k, idx, _] = pcd_tree.search_radius_vector_3d((pcd_.points)[i], 0.007*factor)
        pcd2 = pcd.select_by_index(idx)

        np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        o3d.visualization.draw_geometries([pcd])

        cov_m=np.cov(np.asarray(pcd2.points).transpose())
        cov_m[np.isnan(cov_m)] = 0
        S=np.linalg.svd(cov_m,False,False)  #no full svd nor UV
        k1k2=([S[0],S[1]])
        m_curv=([(S[0]+S[1]/2)])
        g_curv=([(S[0]*S[1]/2)])
        T[0, 3]=(np.asarray(pcd_.points)[i,0])
        T[1, 3]=(np.asarray(pcd_.points)[i,1])
        T[2, 3]=(np.asarray(pcd_.points)[i,2])/100

        print("point",T[0, 3],T[1, 3],T[2, 3])
        R,_=cv2.Rodrigues(PfLEO_OBB.R)

        for r in range(1):
            for p in range(1):
                for y in range(1):
                    Rot_xyz=(np.array([[Rotation(R[0],r,3),Rotation(R[1],p,3),Rotation(R[2],y,3)]]))
                    print("rot: ",Rot_xyz)
                    T[:3, :3],_=cv2.Rodrigues(Rot_xyz)
                    rpy,_=cv2.Rodrigues(T[:3, :3])
                    T[:3, :3]=np.eye(3,3)
                    print( T[:3, :3])
                    x=np.concatenate(([T[0, 3]/np.median(pcd_.points[0],axis=0),T[1, 3]/np.median(pcd_.points[0],axis=0),T[2, 3]/np.median(pcd_.points[0],axis=0)],R[0]-rpy[0],R[1]-rpy[1],R[2]-rpy[2],(pcd_.colors)[i],
                    k1k2,m_curv,g_curv))
                    
                    if len(X)==0:
                        X=np.transpose(np.expand_dims(x,axis=1))
                        Poses=[T]
                        print("100", np.shape(np.array(Poses)))
                    else:
                        X=np.concatenate((X,np.transpose(np.expand_dims(x,axis=1))),axis=0)
                        Poses=np.concatenate((Poses,[T]),axis=0)

            #X=np.concatenate( (np.concatenate((X,np.transpose(np.expand_dims(x,axis=1))),axis=1),x_e),axis=0)

    return np.asarray(Poses), np.asarray(X), pcd_

def ColorAffordance(aff,pcd,color):
    np_colors=np.zeros((len(pcd.points),2))
    np_colors=(np.concatenate((aff,np_colors),axis=1))
    #maxaff=np.argmax(aff)
    #np_colors[maxaff][:]=[0,0,1]

    pcd.colors=o3d.utility.Vector3dVector(np_colors)

    return pcd

def SVD_Principal_Curvature(Pointcloud,radius,factor):
    k1k2=[]
    m_curv=[]
    g_curv=[]
    pcd_tree = o3d.geometry.KDTreeFlann(Pointcloud)

    for i in range(len(Pointcloud.points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d((Pointcloud.points)[i], radius*factor)
        pcd_ = pcd.select_by_index(idx)
        Pointcloud.paint_uniform_color([0,0,0])

        #np.asarray(Pointcloud.colors)[idx[1:], :] = [0, 0, 1]
        #o3d.visualization.draw_geometries([Pointcloud])
        cov_m=np.cov(np.asarray(pcd_.points).transpose())
        cov_m[np.isnan(cov_m)] = 0
        S=np.linalg.svd(cov_m,False,False)  #no full svd nor UV
        k1k2.append([S[0],S[1]])
        m_curv.append([(S[0]+S[1]/2)])
        g_curv.append([(S[0]*S[1]/2)])

    return np.concatenate((np.asarray(k1k2), np.asarray(m_curv), np.asarray(g_curv)),axis=1)

def Extract_Feature(pcd,factor):
    #Non Normalized distance: 
    L=CenterOfPCD(np.asarray(pcd.points))
#curvature features:
    cur=SVD_Principal_Curvature(pcd,0.24,factor)
    x=np.append(cur,L,axis=1)
# Fast Feature
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.1*factor))
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamRadius(radius=0.2*factor))
    
    fph = np.array(np.asarray(fph.data)).T
    fph=np.append(fph,L,axis=1)
    
    x=np.append(cur,fph,axis=1)

    return x

def ColorAffordance(aff,pcd,color):
    np_colors=np.zeros((len(pcd.points),2))
    np_colors=(np.concatenate((aff,np_colors),axis=1))
    #maxaff=np.argmax(aff)
    #np_colors[maxaff][:]=[0,0,1]

    pcd.colors=o3d.utility.Vector3dVector(np_colors)

    return pcd



MSE=[]
MSPE=[]
R2=[]
Var=[]
MAE=[]
PE=[]
Time=[]
for i in range(len(objectlist)):
    if  True:
    #objectlist[i]['semantic class'] == 'Bottle':
        Aff_g=objectlist[i]['partial']['view_2']['label']['grasp']
        xyz=np.asarray(objectlist[i]['partial']['view_2']['coordinate'])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector((xyz))

        x=Extract_Feature(pcd,1)

        aff=reg.predict(x)

        np_colors=np.zeros((len(pcd.points),1))
        aff=np.asarray(aff).reshape((len(pcd.points),1))

        np_colors=(np.concatenate((aff,np_colors,np_colors),axis=1))

        pcd.colors=o3d.utility.Vector3dVector(np_colors)

        #o3d.visualization.draw_geometries([pcd])

        aff=np.asarray(aff).reshape((len(pcd.points),1))
        mSE=sklearn.metrics.mean_squared_error(Aff_g,aff,squared=False)
        r2=sklearn.metrics.r2_score(Aff_g,aff)
        var=sklearn.metrics.explained_variance_score(Aff_g,aff)
        mAE=sklearn.metrics.mean_absolute_error(Aff_g, aff)
        pe=sklearn.metrics.mean_absolute_percentage_error(Aff_g,aff)
        
        Var.append(var)
        MSE.append(mSE)
        R2.append(r2)
        MAE.append(mAE)
        PE.append(pe)

print("Root-mean-square deviation",np.asarray(MSE).mean())
print("R² score, the coefficient of determination",np.asarray(R2).mean())
print("explained_variance_score",np.asarray(Var).mean())
print("Mean absolute error",np.asarray(MAE).mean())
print("average time to compute grasp", np.asarray(Time).mean())
print("average Mean absolute percentage error", np.asarray(PE).mean())





