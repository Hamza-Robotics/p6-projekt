from cProfile import label
import pickle
import open3d as o3d
import numpy as np
from scipy import stats
import itertools
import random
import time

def CenterOfPCD(PCD):
    m_xyz=np.mean(PCD,axis=0)
    L=np.linalg.norm(m_xyz-PCD,axis=1).reshape((len(PCD),1))
    #L=(L-min(L))/(max(L)-min(L))

    return L

pickle_file = 'C:\\data_for_learning\\full_shape_train_data.pkl' ### Write path for the full_shape_val_data.pkl file ###

with open(pickle_file, 'rb') as f:
    Data = pickle.load(f)

def DataDeriver(data,n_,f_,n_nei,f_nei):
    j = 0
    #print(data[0]['full_shape']['label'])#['coordinate'])
    objectlist=[]
    while(1):
        pass
        break
    for i in range(len(data)):
        if data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle'  or data[i]['semantic class'] == 'Bowl' :
            objectlist.append(data[i])
        

    x=[]
    y=[]
    t=[]
    labels=np.asarray([])
    for i in range(len(objectlist)):
    #        if i  in [0,362,366,377,380,382,329,399,413,415.407,421,424,432,433,434,436,442,456,469,475,480,483,489,490,492,
     #       501,511,524,536,538,543,547,550,552,553,561,568,571,571,577,583,588,591,616,617,619,629]:
      #          print(i)
        if True:

            #377 380 382 456 475
            xyz=np.asarray(objectlist[i]['full_shape']['coordinate'])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector((xyz))

            if objectlist[i]['semantic class'] == 'Bowl'  :
                Aff_v1=objectlist[i]['full_shape']['label']['pourable']

            if objectlist[i]['semantic class'] == 'Bottle':
                Aff_v1=objectlist[i]['full_shape']['label']['wrap_grasp']
            else:  
                Aff_v1=objectlist[i]['full_shape']['label']['grasp']

            Aff_v2=objectlist[i]['full_shape']['label']['wrap_grasp']
            np_colors=np.zeros((len(pcd.points),1))
            np_colors=(np.concatenate((Aff_v1,np_colors,np_colors),axis=1))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector((xyz))
            pcd.colors=o3d.utility.Vector3dVector(np_colors)

            diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
            pp=1
            for l in range(1):
                radius = diameter * 100



                pp=pp*(-1)
                if objectlist[i]['semantic class'] == 'Bottle': 
                    camera=[diameter,0,-diameter]
                else:
                    camera = [diameter,-diameter,-diameter]

                _, pt_map = pcd.hidden_point_removal(camera, radius)
                pcd2 = pcd.select_by_index(pt_map)



                be=time.time()


                pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*n_, max_nn=n_nei))
                #o3d.visualization.draw_geometries([pcd2])

                fph=o3d.pipelines.registration.compute_fpfh_feature(pcd2, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*f_, max_nn=f_nei))

                
                L=CenterOfPCD(np.asarray(pcd2.points))


                fph = np.array(np.asarray(fph.data)).T
                #x.append(fph)
                fph=np.append(fph,L,axis=1)
                for k in range(len(np.array(pcd2.points))):
                    x.append(fph[k])
                #labels=[objectlist[i]['full_shape']['label']['grasp'], objectlist[i]['full_shape']['label']['wrap_grasp'], objectlist[i]['full_shape']['label']['contain']]
                #labels = np.reshape(labels,(np.shape(labels)[0],np.shape(labels)[1]))
                if i==0:
                    y=[np.asarray(np.asarray(pcd2.colors)[:,:1])]
                    t=[time.time()-be]

                else:    
                    y=np.append(y,([np.asarray(np.asarray(pcd2.colors)[:,:1])]),axis=1)
                    t.append(time.time()-be)
                


        if i==len(objectlist)-1:
            print("i:::",i)          #print(np.shape([np.asarray(np.asarray(pcd.colors)[:,:1]),np.asarray(np.asarray(pcd.colors)[:,1:2])]))
    print("time",np.average(np.asarray(t)))
    return x,y


n_v=[3] #5 normals
f_v=[5] #feature
n_n=[70] #max niegbourh
f_n=[160]

c = list(itertools.product(n_v, f_v,n_n,f_n))
c=np.asarray(c)

print(c[0][0])
corp=0
best_cor=0
best_C=[]
for i in range(len(c)):
    x,y=DataDeriver(Data,c[i][0],c[i][1],c[i][2],c[i][3])
    x=np.asarray(x)
    y=np.asarray(y).reshape(np.shape(y)[1],np.shape(y)[0])
    cor=np.sum(np.absolute(stats.spearmanr(x, y)))

    if cor>corp:
        corp=cor
        best_cor=cor
        best_C=c







print(np.shape(y))
print(np.shape(x))

np.save('C:\\data_for_learning\\x_values.npy', x)
np.save('C:\\data_for_learning\\y_values.npy', y)
np.save


