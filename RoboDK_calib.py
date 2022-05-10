import cv2
import numpy as np
import open3d as o3d
import json
import pyrealsense2 as rs
import time
from robolink import *    # API to communicate with RoboDK
from robodk import *      # robodk robotics toolbox

# Any interaction with RoboDK must be done through RDK:
RDK = Robolink()

# Select a robot (popup is displayed if more than one robot is available)
robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception('No robot selected or available')

with open("CameraSetup.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))
##Startting camera
pipeline = rs.pipeline()
cfg = pipeline.start() # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0)
intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,intr.fx,intr.fy,intr.ppx,intr.ppy)
rs.start_capture(True)  # true: start recording with capture  
im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the
img=np.asarray(im_rgbd.color)
mtx=(intrinsic.intrinsic_matrix)
n=np.array([])

def CameraPose(img,mtx):
#solvepnp from chessboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)*20

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,9),None)
    #solvepnp with factory calibration

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx,None)


        return ret, rvecs, tvecs
    else:
        return False, [],[]

R_gripper2base=[]
t_gripper2base=[]
R_target2cam=[]
t_target2cam=[]

i=0
while (i<10):
    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the
    img=np.asarray(im_rgbd.color)

    ret,rvec,tvec=CameraPose(img,mtx)
    pose=np.asarray(robot.Pose().Rows())
    rot=pose[0:3,0:3]
    trans=np.reshape(pose[0:3,3],(3,1))
    if ret==True:
        R_gripper2base.append(rot)
        t_gripper2base.append(trans)
        R_target2cam.append(rvec)
        t_target2cam.append(tvec)
        i=i+1
    #time.sleep(0.4)


R_gripper2base=  np.asarray(R_gripper2base)
t_gripper2base=  np.asarray(t_gripper2base)
R_target2cam=  np.asarray(R_target2cam)
t_target2cam=  np.asarray(t_target2cam)

R,T=cv2.calibrateHandEye(R_gripper2base,t_gripper2base,R_target2cam,t_target2cam)
print(R)
print(T)
