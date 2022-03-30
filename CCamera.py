from ctypes.wintypes import PLCID
import json
from pickle import FALSE
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import msvcrt

from tomlkit import inline_table


with open("CameraSetup.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0)
rs.start_capture(True)  # true: start recording with capture


def visualizer(rgb,Depth,hasrun=[]):
    
    if not hasrun:
        hasrun.append("second time")
        plt.ion()
        DEPTH = plt.subplot(1,2,1)
        RGB = plt.subplot(1,2,2)
        global im1
        global im2
        im2=RGB.imshow(rgb)
        im1=DEPTH.imshow(Depth)


    else:
        im2.set_data(rgb)
        im1.set_data(Depth)
        plt.show()
        plt.pause(0.001)

im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
cam = o3d.camera.PinholeCameraIntrinsic()
intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[231.7225,0,161.8237],[0,229.9110,80.4734],[0,0,1.0000]])))
pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd,intrinsic)
pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def DefinePointCloud(img,Voxel_Choise,intrinsicMatrix):
    PCL = o3d.t.geometry.PointCloud.create_from_rgbd_image(img,intrinsic)
    PCL=o3d.t.geometry.PointCloud.to_legacy(PCL)
    if Voxel_Choise==True:
        voxel_down_pcd = PCL.voxel_down_sample(voxel_size=0.02)
        voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        return voxel_down_pcd

    else: 
        return PCL

def Segmentation(pcl):
    plane_model, inliers = pcl.segment_plane(distance_threshold=0.30,
                                         ransac_n=3,
                                         num_iterations=2000)       
    inlier_cloud = pcl.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcl.select_by_index(inliers, invert=True)

    return inlier_cloud,outlier_cloud

def Clustering(pcl,eps,min_points):
    labels = np.array(pcl.cluster_dbscan(eps, min_points,False))
    #dbscan labeller alle point clouds. Man finder den point cloud med højeste label. En cluster er alle de points med samme label. 
    max_label = labels.max()
    Clus=[]


    for i in range(max_label+1):
        id=np.where(labels==i)[0]
        pcl_i=pcl.select_by_index(id)
        Clus.append(pcl_i)

    return Clus



im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
PCL=DefinePointCloud(im_rgbd,True,intrinsic)
inc,PCL= Segmentation(PCL)
#eps lav en cirkel med radius af 30 cm hvis der er 425 obj i så er det en cluster. 
Clus=Clustering(PCL,eps=0.2, min_points=202)
obs=[]
for i in range(len(Clus)):
   # print(type(Clus[i]))
   obs.append(Clus[i])

o3d.visualization.draw_geometries(obs)
#print(np.size(np.asarray(PCL.points)))                        
# Features from each object 
#visualizer(im_rgbd.depth,im_rgbd.color)


#if msvcrt.kbhit():
#    if msvcrt.getwche() == 'c':
     #   break






plt.ioff() # due to infinite loop, this gets never called.
plt.close()
