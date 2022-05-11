import cv2
import numpy as np
from asyncio.windows_events import NULL


img=cv2.imread('img2.png')

mtx=np.asarray([[420.12387085,0.,421.87664795],[0.,420.12387085,241.06109619],[0.,0.,1.]])

mtx1=np.asarray([[231.7225,0,161.8237],[0,229.9110,80.4734],[0,0,1.0000]])

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9*7,3), np.float32)

objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)

gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (7,9),None)

#solvepnp with factory calibration

if ret == True:

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    # Find the rotation and translation vectors.

    ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx,NULL)

    print("norm:\n",np.linalg.norm(tvecs))

    print("vector:\n",tvecs)

#solvepnp with matlab calibration

if ret == True:

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    # Find the rotation and translation vectors.

    ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx1,NULL)

    print("norm:\n",np.linalg.norm(tvecs))

    print("vector:\n",tvecs)