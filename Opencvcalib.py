import numpy as np
import cv2 as cv
import open3d as o3d
import pyrealsense2 as rs
import json
import time


with open("CameraSetup.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)*0.021
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


pipeline = rs.pipeline()
cfg = pipeline.start() # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0)
o3d.t.io.RealSenseSensor.list_devices()

#intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,intr.fx,intr.fy,intr.ppx,intr.ppy)

rs.start_capture(True)  # true: start recording with capture  
im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the
img=np.asarray(im_rgbd.color)

i=0
while(i<50):
    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the
    img=np.asarray(im_rgbd.color)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        i=i+1
        time.sleep(3)
        print(i)
        # Draw and display the corners


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.save("Calibration__Data\\intrinicmat640xo480.npy",mtx)
np.save("Calibration__Data\\dist.npy",dist)

print(ret)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

print(mtx)