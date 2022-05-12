
from matplotlib.transforms import Affine2D
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import copy



from mpl_toolkits import mplot3d

pcd1=np.load('pcd.npy')


pcd_color=np.load('pcd_c.npy')

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector((pcd1))
nps=np.zeros(np.shape(pcd_color[:,0]))
color=[pcd_color[:,0],nps,nps]

print(np.shape(np.asanyarray(color)))
pcd.colors=o3d.utility.Vector3dVector(np.transpose(color))
#pcd_v = pcd.voxel_down_sample(voxel_size=0.0)

pcd_v=pcd
y=np.asarray(pcd_v.points)
x=np.asanyarray(pcd_v.colors)[:,0]
print(np.shape(np.asanyarray(pcd_v.colors)))
scipy.io.savemat('test.mat', dict(x=x,y=y))

pcd=np.asarray(pcd_v.points)


print("compute the svd")


#uu, dd, vv = np.linalg.svd((pcd))
#ts = vv[0] * np.mgrid[-np.max(pcd)*2:np.max(pcd):20j][:, np.newaxis]


line = o3d.geometry.PointCloud()
#line.points = o3d.utility.Vector3dVector(((linepts)))
line.paint_uniform_color([1, 0, 0])

print(np.shape(np.asanyarray(pcd_v.points)))
aff_m=np.argmax(np.asanyarray(pcd_v.colors)[:,0])
aff=pcd_v.select_by_index([aff_m])
aff.paint_uniform_color([1, 1, 0])

print("p=",np.asarray(aff.points))
PfLEO_OBB = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_v.points)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
T = np.eye(4)
T[:3, :3]= PfLEO_OBB.R
print(np.shape(np.asanyarray(aff.points)))
T[0, 3]=np.asanyarray(aff.points)[0,0]
T[1, 3]=np.asanyarray(aff.points)[0,1]
T[2, 3]=np.asanyarray(aff.points)[0,2]
mesh_t = copy.deepcopy(mesh).transform(T)
mesh_t.scale(0.1, center=mesh_t.get_center())



#pcd2.points=o3d.utility.Vector3dVector(pcd)
o3d.visualization.draw_geometries([pcd_v,aff,mesh_t])
#ax.plot3D(*linepts.T)


x, y, z = zip(*pcd)


# make and show plot
#ax.scatter(x,y,z)
#plt.show()