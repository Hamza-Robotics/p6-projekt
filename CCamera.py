import json
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
    plane_model, inliers = pcl.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)       
    inlier_cloud = pcl.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcl.select_by_index(inliers, invert=True)

    return inlier_cloud,outlier_cloud

while True:

    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
    PCL=DefinePointCloud(im_rgbd,True,intrinsic)
    inc,outc= Segmentation(PCL)

    o3d.visualization.draw_geometries([outc],
                                zoom=0.8,
                                front=[-0.4999, -0.1659, -0.8499],
                                lookat=[2.1813, 2.0619, 2.0999],
                                up=[0.1204, -0.9852, 0.1215])
                                         
    # Features from each object 
    #visualizer(im_rgbd.depth,im_rgbd.color)

    if msvcrt.kbhit():
        if msvcrt.getwche() == 'c':
            break






plt.ioff() # due to infinite loop, this gets never called.
plt.close()
