import pyrealsense2 as rs
import numpy as np
import open3d as o3d 
import json
import cv2
import time


def setup():
    with open("CameraSetup.json") as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))
    rs = o3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0)
    rs.start_capture(True)  # true: start recording with capture 
    #mtx=np.load("RealSenseCameraParmas.npy") 
    #mtx=np.load("RealSenseCameraParams\\RealSenseCameraParmas1280x720.npy")
    mtx=np.load("Calibration__Data\\intrinicmat640xo480.npy")
    return rs,mtx
rs,mtx=setup()


def CameraPose(img,mtx):
#solvepnp from chessboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)*0.022

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

def rot_params_rv(rvecs):
    from math import pi,atan2,asin
    R = rvecs
    roll = 180*atan2(-R[2][1], R[2][2])/pi
    pitch = 180*asin(R[2][0])/pi
    yaw = 180*atan2(-R[1][0], R[0][0])/pi
    rot_params= [roll,pitch,yaw]
    return rot_params
while True:
    time.sleep(4)
    im_rgbd = rs.capture_frame(True, True)  
    img=np.asanyarray(im_rgbd.color)
    ret,rvec,tvec=CameraPose(img,mtx)

    if ret==True:  
        print(np.shape(img))
        print("x: ",tvec[0],"y: ",tvec[1],"z: ",tvec[2], "norm", np.linalg.norm(tvec))
        print("\n\n\n\n\n\n\n\n\n\n\n\n")
        print(rvec)
        R,_=cv2.Rodrigues(rvec)
        r=rot_params_rv(rvec)
        #print("Rx: ",r[0],"Ry: ",r[1],"Rz: ",r[2])

 




