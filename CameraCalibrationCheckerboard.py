import numpy as np
import open3d as o3d
import json
import pyrealsense2 as rs
import time
import sys
import urx
import math3d as m3d

import cv2 as cv

while True:
    try:
        time.sleep(3)
        rob = urx.Robot("172.31.1.115")
    except:
        rob.stop()
    else:
        break
a=0.6
v=0.5

with open("CameraSetup.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))
##Startting camera
pipeline = rs.pipeline()
cfg = pipeline.start() # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0)
#intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,intr.fx,intr.fy,intr.ppx,intr.ppy)
#intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,317.9628,320.223,216.6219,114.8350)
#intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,976.23415007,980.51695051, 695.12887163,226.97863859)

rs.start_capture(True)  # true: start recording with capture  
im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the
img=np.asarray(im_rgbd.color)
n=np.array([])



criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)*0.0225
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

import pickle
pickle_file = "poseList.pickle" ### Write path for the poseList.pickle file ###
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
i=0
while (i<30):
    if i !=100:
        rob.movej(data[i],a,v)
        time.sleep(1)
        im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the
        time.sleep(2)
        img=np.asarray(im_rgbd.color)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,9), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            
            print(i)
    i=i+1
        # Draw and display the corners
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


np.save("Calibration__Data\\intrinicmat1280x720.npy",mtx)
np.save("Calibration__Data\\dist.npy",dist)

print(ret)
print(mtx)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

rob.stop()
rob.close()
sys.exit()


