import open3d as o3d
import numpy as np
gg = o3d.geometry.PointCloud()
np.random.normal(loc=0.0, scale=1.0, size=None)

points=[]
for i in range(1000):
    point=[[10,+1/10*i,0+i]]
    points.append(point)

print(np.random.normal(loc=0, scale=4.0, size=None))
print(np.shape(np.array(points).reshape(1000,3)))
gg.points = o3d.utility.Vector3dVector(np.array(points))
gg.paint_uniform_color([0,1,1])

print(np.max(np.asarray(gg.points)[:,2]))