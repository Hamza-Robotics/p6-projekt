import pyrealsense2 as rs
import cv2
import open3d as o3d
import json
import numpy as np
import time
 

if False:
    with open("CameraSetup.json") as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

    pipeline = rs.pipeline()

    cfg = pipeline.start() # Start pipeline and get the configuration it found
    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

    rs = o3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0)
    rs.start_capture(True)  # true: start recording with capture 
    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
    imgd_s=["imgd1","imgd2","imgd3"]
    img_s=["img1","img2","img3"]
    for i in range(len((imgd_s))):
        im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them

        imgd=np.asarray(im_rgbd.depth)
        img=np.asarray(im_rgbd.color)
        print("sleeping for 5 minutes")
        cv2.imwrite('C:/data_for_learning/'+imgd_s[i]+'.png', imgd)
        cv2.imwrite('C:/data_for_learning/'+img_s[i]+'.png', img)
        time.sleep(5)


imgd_s=["imgd1","imgd2","imgd3"]
img_s=["img1","img2","img3"]
for i in range(len(img_s)):
    kernel = np.ones((1,1),np.uint8)
    kernel2 = np.ones((13,13),np.uint8)
    img=cv2.imread('C:/data_for_learning/'+img_s[i]+'.png')
    depth=cv2.imread('C:/data_for_learning/'+imgd_s[i]+'.png')       

    img1= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img1=cv2.GaussianBlur(img1,(13,13),0)

    img_t=cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    img_t=cv2.morphologyEx(img_t,cv2.MORPH_OPEN,kernel)
    img_t=cv2.morphologyEx(img_t,cv2.MORPH_CLOSE,kernel2)
    #img_t=cv2.morphologyEx(img_t,cv2.MORPH_DILATE,kernel2)

    contours, hierarchy = cv2.findContours(img_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                            



    
    for l in range(len(contours)):
        cnt = contours[l]
        print(cv2.arcLength(cnt,True))
        print("cont",  cv2.contourArea(cnt))

        if cv2.contourArea(cnt)>1000 and cv2.arcLength(cnt,True)<2400:
            
            cv2.drawContours(img, contours,l, (0,255,0),thickness=cv2.FILLED)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

                                    
            cv2.imshow("Img",img)
            cv2.waitKey(0)
                    
            print("cont",  cv2.contourArea(cnt))
            #cv2.imshow("image", img2)   
            #cv2.waitKey(0)