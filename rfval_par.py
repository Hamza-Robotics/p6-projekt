import pickle 
import numpy as np
import open3d as o3d
print("hello")
pickle_file = "C:\\data_for_learning\\partial_val_data.pkl" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
pickle_file = "C:\\data_for_learning\\RegressionGrasp2.pickle" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)



def CenterOfPCD(PCD):
    m_xyz=np.mean(PCD,axis=0)
    L=np.linalg.norm(m_xyz-PCD,axis=1).reshape((len(PCD),1))
    #L=(L-min(L))/(max(L)-min(L))

    return L
def SVD_Principal_Curvature(Pointcloud,radius):
    k1k2=[]
    m_curv=[]
    g_curv=[]
    pcd_tree = o3d.geometry.KDTreeFlann(Pointcloud)

    for i in range(len(Pointcloud.points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        pcd_ = pcd.select_by_index(idx)
        cov_m=np.cov(np.asarray(pcd_.points).transpose())
        cov_m[np.isnan(cov_m)] = 0
        S=np.linalg.svd(cov_m,False,False)  #no full svd nor UV
        k1k2.append([S[0],S[1]])
        m_curv.append([(S[0]+S[1]/2)])
        g_curv.append([(S[0]*S[1]/2)])

    return np.concatenate((np.asarray(k1k2), np.asarray(m_curv), np.asarray(g_curv)),axis=1)

objectlist=[]
for i in range(len(data)):
    if (data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle' or data[i]['semantic class'] == 'Bowl' or data[i]['semantic class'] == 'Mug'):
        objectlist.append(data[i])

#for i in range(len(objectlist)):
def Extract_Feature(pcd):
#Non Normalized distance: 
    L=CenterOfPCD(np.asarray(pcd.points))
#curvature features:
    cur=SVD_Principal_Curvature(pcd,0.07)
    x=np.append(cur,L,axis=1)
    print("prin1t")
# Fast Feature
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.3))
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamRadius(radius=0.06))
    fph = np.array(np.asarray(fph.data)).T
    fph=np.append(fph,L,axis=1)
    
    x=np.append(cur,fph,axis=1)

    return x

for i in range(30):

    #Aff_v1=objectlist[i]['partial']['view_2']['label']['grasp']
    xyz=np.asarray(objectlist[i]['partial']['view_2']['coordinate'])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((xyz))

    x=Extract_Feature(pcd)

    aff=reg.predict(x)



    np_colors=np.zeros((len(pcd.points),1))
    aff=np.asarray(aff).reshape((len(pcd.points),1))

    np_colors=(np.concatenate((aff,np_colors,np_colors),axis=1))

    pcd.colors=o3d.utility.Vector3dVector(np_colors)

    o3d.visualization.draw_geometries([pcd])



