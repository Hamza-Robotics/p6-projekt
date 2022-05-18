import open3d as o3d
import pickle
import numpy as np


pickle_file = 'C:\\data_for_learning\\partial_train_data.pkl' ### Write path for the full_shape_val_data.pkl file ###

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

objectlist=[]
for i in range(len(data)):
    if data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle':
        objectlist.append(data[i])



x=[]
y=[]

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

        if i==1:
            np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
            #o3d.visualization.draw_geometries([pcd])
        cov_m=np.cov(np.asarray(pcd_.points).transpose())
        cov_m[np.isnan(cov_m)] = 0
        S=np.linalg.svd(cov_m,False,False)  #no full svd nor UV
        k1k2.append([S[0],S[1]])
        m_curv.append([(S[0]+S[1]/2)])
        g_curv.append([(S[0]*S[1]/2)])

    return np.concatenate((np.asarray(k1k2), np.asarray(m_curv), np.asarray(g_curv)),axis=1)

for i in range(len(objectlist)):
    m=0
    for j in (objectlist[i]['partial']):
        if (str(j) in ["view_0","view_1"]):
            if True:
                print(i/(len(objectlist)))
                Aff_v1=objectlist[i]['partial'][j]['label']['grasp']
                xyz=np.asarray(objectlist[i]['partial'][j]['coordinate'])  
                from scipy.io import savemat
                mdic={"pointcloud":xyz}
                savemat("matlab_matrix.mat", mdic)  
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector((xyz))
                np_colors=np.zeros((len(pcd.points),1))
                np_colors=(np.concatenate((Aff_v1,np_colors,np_colors),axis=1))
                pcd.colors=o3d.utility.Vector3dVector(np_colors)
                pcd = pcd.voxel_down_sample(voxel_size=0.0001)



                #Non Normalized distance: 
                L=CenterOfPCD(np.asarray(pcd.points))

                #curvature features:
                cur=SVD_Principal_Curvature(pcd,0.14)


                #Fast Features
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.1))
                #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
                fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamRadius(radius=0.2))
                fph = np.array(np.asarray(fph.data)).T
                fph=np.append(fph,L,axis=1)
                m=m+1


                if i==0:
                    y=[np.asarray(pcd.colors)[:,0]]
                    x=np.append(cur,fph,axis=1)
                    print(np.shape(x))
                else:    
                    y=np.append(y,([np.asarray(np.asarray(pcd.colors)[:,0])]),axis=1)
                    x=np.append(x,np.append(cur,fph,axis=1),axis=0)
                print("x: ",np.shape(x)," y: ",np.shape(y))




x=(np.squeeze(np.asarray(x)))
y=np.expand_dims(np.squeeze(np.asarray(y)),axis=1)

print("x: ",np.shape(x)," y: ",np.shape(y))

np.save('C:\\data_for_learning\\x_values_c.npy', x)
np.save('C:\\data_for_learning\\y_values_c.npy', y)