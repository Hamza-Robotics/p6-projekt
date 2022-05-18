import numpy as np
import open3d as o3d
import urx
import json
import sys

Method_Tsai=1
Method_Andref=2
Method_Dadnilist=3
Method_Horud=4
Method_Park=5
def calculateT2B(world2camMat,meth):
    if meth==1:
        cam2gripperMat = np.load("Calibration__Data\\HandEyeTransformation.npy")
    if meth==2: 
        cam2gripperMat = np.load("Calibration__Data\\HandEyeTransformation_andereff.npy")
    if meth==3: 
        cam2gripperMat = np.load("Calibration__Data\\HandEyeTransformation_DANIILIDIS.npy")
    if meth==4: 
        cam2gripperMat = np.load("Calibration__Data\\HandEyeTransformation_HORAUD.npy")
    if meth==5:
        cam2gripperMat = np.load("Calibration__Data\\HandEyeTransformation_park.npy")

    
    gripper2baseMat = rob.get_pose()

    gripper2baseMat = gripper2baseMat.get_matrix()

    world2base = (gripper2baseMat) * (cam2gripperMat) * (world2camMat)

    return world2base
def RemoveBackGround(rgbd,d):
    a=np.asarray(im_rgbd.depth)

    a=(a < d*1000)*a

    im=np.asarray(im_rgbd.color)

    Im_Rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(im), o3d.geometry.Image(a),convert_rgb_to_intensity=False)

    return Im_Rgbd

with open("CameraSetup.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

rob = urx.Robot("172.31.1.115")
print("pose:",rob.get_pose())


rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0)
rs.start_capture(True)  # true: start recording with capture  
mtx=np.load("RealSenseCameraParams\\RealSenseCameraParmas640x480.npy")
intrinsic=o3d.camera.PinholeCameraIntrinsic(640,480 ,mtx[0][0],mtx[1][1],mtx[0][2],mtx[1][2])
o3d.t.io.RealSenseSensor.list_devices()
im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the
im_rgbd=RemoveBackGround(im_rgbd,1)
pcd=o3d.geometry.PointCloud.create_from_rgbd_image(im_rgbd,intrinsic)

plane_model, inliers = pcd.segment_plane(distance_threshold=0.004,
                                         ransac_n=400,
                                         num_iterations=10000)


o3d.visualization.draw_geometries([pcd  ])

rob.stop()
rob.close()
sys.exit()
