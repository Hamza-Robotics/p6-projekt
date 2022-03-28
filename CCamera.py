import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import msvcrt


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



while True:
    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them


    #print(o3d.camera.PinholeCameraIntrinsic)
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image( im_rgbd, o3d.camera.PinholeCameraIntrinsic(
            #o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    
    # Covert image into point cloud.
    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #o3d.visualization.draw_geometries([pcd], zoom=0.5)
    #Filtering


    # Plane Object Segmentation

    

    # Features from each object 
    visualizer(im_rgbd.depth,im_rgbd.color)
    if msvcrt.kbhit():
        if msvcrt.getwche() == 'c':
            break






plt.ioff() # due to infinite loop, this gets never called.
plt.close()
