import cv2
from cv2 import waitKey
from matplotlib.pyplot import rgrids
import numpy as np
import open3d as o3d
import json
import pyrealsense2 as rs
import pickle
import time
import sys
import urx
import math3d as m3d

while True:
    try:
        time.sleep(3)
        rob = urx.Robot("172.31.1.115")
    except:
        rob.stop()
    else:
        break

a=0.4
v=0.2
ori = rob.get_orientation()



with open("CameraSetup.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

pickle_file = "poseList.pickle" ### Write path for the poseList.pickle file ###
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

##Startting camera
pipeline = rs.pipeline()
cfg = pipeline.start() # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0)
#intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,intr.fx,intr.fy,intr.ppx,intr.ppy)
mtx=np.load("Calibration__Data\\intrinicmat640xo480.npy")
rs.start_capture(True)  # true: start recording with capture  
im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the
img=np.asarray(im_rgbd.color)
n=np.array([])


def rot_params_rv(rvecs):
    from math import pi,atan2,asin
    R = cv2.Rodrigues(rvecs)[0]
    roll = atan2(-R[2][1], R[2][2])
    pitch = asin(R[2][0])
    yaw = atan2(-R[1][0], R[0][0])
    rot_params= [roll,pitch,yaw]
    return rot_params


def CameraPose(img,mtx):
#solvepnp from chessboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)*0.021

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,9),None)
    #solvepnp with factory calibration

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx,None)

        R,_=cv2.Rodrigues(rvecs)

        return ret, R, tvecs
    else:
        return False, [],[]

R_gripper2base=[]
t_gripper2base=[]
R_target2cam=[]
t_target2cam=[]

i=0
while (i<len(data)):
    
    rob.movej(data[i],a,v)
    time.sleep(2)
    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the
    img=np.asarray(im_rgbd.color)
    time.sleep(2)

    ret,rvec,tvec=CameraPose(img,mtx)
    #pose=np.asarray()
    #rot=pose[0:3,0:3]
    #trans=np.reshape(pose[0:3,3],(3,1))
    ori = rob.get_orientation()
    pos = rob.get_pos()
    
    rot=np.asarray([ori[0],ori[1],ori[2]])
    trans=np.asarray([pos[0], pos[1], pos[2]])
    l=1
    if ret==True:
        print(l)
        l=1+l
        R_gripper2base.append(rot)
        t_gripper2base.append(trans)
        R_target2cam.append(rvec)
        t_target2cam.append(tvec)


        #print("translation error:\n",(trans)-(tvec),"\n")
        #print("rotation error:\n",rot-trans,"\n")
    i=i+1
    #time.sleep(0.4)


R_gripper2base=  np.asarray(R_gripper2base)
t_gripper2base=  np.asarray(t_gripper2base)
R_target2cam=  np.asarray(R_target2cam)
t_target2cam=  np.asarray(t_target2cam)

print(np.shape(t_target2cam))

#print(type(R_gripper2base),R_gripper2base)
#print(type(t_gripper2base),t_gripper2base)
#print(type(t_target2cam),t_target2cam)
#print(type(R_target2cam),R_target2cam)
#print("shape",np.shape(R_gripper2base))
#print("shape",np.shape(t_gripper2base))

#print(R_gripper2base)


np.save("R_gripper2base.npy",R_gripper2base)
np.save("t_gripper2base.npy",t_gripper2base)
np.save("R_target2cam.npy",R_target2cam)
np.save("t_target2cam.npy",t_target2cam)

R,t=cv2.calibrateHandEye(R_gripper2base,t_gripper2base,R_target2cam,t_target2cam,method=cv2.CALIB_HAND_EYE_TSAI)
T = np.eye(4)
T[:3, :3]= R
T[0, 3]=t[0]
T[1, 3]=t[1]
T[2, 3]=t[2]
print("CALIB_HAND_EYE_TSAI\n\n")
print(T)
print("\n\n")

np.save("Calibration__Data\\HandEyeTransformation",T)



#print(R_gripper2base)
R,t=cv2.calibrateHandEye(R_gripper2base,t_gripper2base,R_target2cam,t_target2cam,method=cv2.CALIB_HAND_EYE_PARK)
T = np.eye(4)
T[:3, :3]= R
T[0, 3]=t[0]
T[1, 3]=t[1]
T[2, 3]=t[2]
print("cv2.CALIB_HAND_EYE_PARK = 1,\n\n")
print(T)
print("\n\n")
np.save("Calibration__Data\\HandEyeTransformation_park",T)


R,t=cv2.calibrateHandEye(R_gripper2base,t_gripper2base,R_target2cam,t_target2cam,method= cv2.CALIB_HAND_EYE_ANDREFF)
T = np.eye(4)
T[:3, :3]= R
T[0, 3]=t[0]
T[1, 3]=t[1]
T[2, 3]=t[2]
print("cv2.CALIB_HAND_EYE_ANDREFF,\n\n")
print(T)
print("\n\n")
np.save("Calibration__Data\\HandEyeTransformation_andereff",T)



R,t=cv2.calibrateHandEye(R_gripper2base,t_gripper2base,R_target2cam,t_target2cam,method= cv2.CALIB_HAND_EYE_HORAUD)
T = np.eye(4)
T[:3, :3]= R
T[0, 3]=t[0]
T[1, 3]=t[1]
T[2, 3]=t[2]
print("cv2.CALIB_CALIB_HAND_EYE_HORAUD,\n\n")
print(T)
print("\n\n")
np.save("Calibration__Data\\HandEyeTransformation_HORAUD",T)


R,t=cv2.calibrateHandEye(R_gripper2base,t_gripper2base,R_target2cam,t_target2cam,method= cv2.CALIB_HAND_EYE_HORAUD)
T = np.eye(4)
T[:3, :3]= R
T[0, 3]=t[0]
T[1, 3]=t[1]
T[2, 3]=t[2]
print("cv2.HandEyeTransformation_DANIILIDIS,\n\n")
print(T)
print("\n\n")
np.save("Calibration__Data\\HandEyeTransformation_DANIILIDIS",T)

rob.stop()
rob.close()
sys.exit()


 