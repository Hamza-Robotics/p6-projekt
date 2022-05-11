import cv2
import numpy as np
import open3d as o3d
import json
import pyrealsense2 as rs
import time
import sys
import urx
import time

rob = urx.Robot("172.31.1.115")
a=0.4
v=0.2

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

def moveIngremental(begin, vertical, i):
    print("moving robot")
    if vertical and begin:
        rob.movej([0.7853862047195435, -1.515883747731344, 2.183744430541992, 0.028925776481628418, 1.7645397186279297, 1.702071189880371], a, v, wait=True, relative=False)
    elif vertical:
        rob.movel((0, 0, 0.15, np.pi/(40+i*10), -np.pi/(40+i*10), 0), a, v, wait=True, relative=True)
    elif begin:
        rob.movej([0.11325842142105103, -0.38410789171327764, 0.6964402198791504, 1.9883853197097778, 0.7105001211166382, 0.6198690533638], a, v, wait=True, relative=False)
    else:
        rob.movel((0.01, -0.01, 0, np.pi/20, np.pi/20, 0), a, v, wait=True, relative=True)
    time.sleep(4)

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
while (i<15):
    if (i < 9):
        vert = False
    else:
        vert = True

    if (i==0 or i==9):
        moveIngremental(True, vert, i)
    else:
        moveIngremental(False, vert, i-9)

    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the
    img=np.asarray(im_rgbd.color)

    ret,rvec,tvec=CameraPose(img,mtx)
    #pose=np.asarray()
    #rot=pose[0:3,0:3]
    #trans=np.reshape(pose[0:3,3],(3,1))
    ori = rob.get_orientation()
    pos = rob.get_pos()
    rot=np.asarray([ori[0],ori[1],ori[2]])
    trans=np.asarray([pos[0], pos[1], pos[2]])
    if ret==True:
        R_gripper2base.append(rot)
        t_gripper2base.append(trans)
        R_target2cam.append(rvec)
        t_target2cam.append(tvec)
        print("Yay")
    i=i+1
    #time.sleep(0.4)


R_gripper2base=  np.asarray(R_gripper2base)
t_gripper2base=  np.asarray(t_gripper2base)
R_target2cam=  np.asarray(R_target2cam)
t_target2cam=  np.asarray(t_target2cam)

#print(type(R_gripper2base),R_gripper2base)
#print(type(t_gripper2base),t_gripper2base)
#print(type(t_target2cam),t_target2cam)
#print(type(R_target2cam),R_target2cam)
#print("shape",np.shape(R_gripper2base))
#print("shape",np.shape(t_gripper2base))

#print(R_gripper2base)
R,T=cv2.calibrateHandEye(R_gripper2base,t_gripper2base,R_target2cam,t_target2cam)
print(R)
print(T)
