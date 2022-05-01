#from contextlib import closing
#import p6
import json
from cv2 import cvtColor
import pyrealsense2 as rs
#import time
import open3d as o3d
import numpy as np
import cv2
import msvcrt
import pickle

def CenterOfPCD(PCD):
    m_xyz=np.mean(PCD,axis=0)
    L=np.linalg.norm(m_xyz-PCD,axis=1).reshape((len(PCD),1))
    L=(L-min(L))/(max(L)-min(L))
    return L

pickle_file = "C:\\data_for_learning\\RegressionGrasp.pickle" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)


if False:
    pipeline = rs.pipeline()

    cfg = pipeline.start() # Start pipeline and get the configuration it found
    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

    #intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[231.7225,0,161.8237],[0,229.9110,80.4734],[0,0,1.0000]])))
    intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])))
    with open("CameraSetup.json") as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

    rs = o3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0)
    rs.start_capture(True)  # true: start recording with capture  

    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the

    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
    img=np.asarray(im_rgbd.color)
pipeline = rs.pipeline()

cfg = pipeline.start() # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

#intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[231.7225,0,161.8237],[0,229.9110,80.4734],[0,0,1.0000]])))
#intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])))
intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,intr.fx,intr.fy,intr.ppx,intr.ppy)

def run():


    with open("CameraSetup.json") as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

    rs = o3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0)
    rs.start_capture(True)  # true: start recording with capture  

    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the

    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
    img=np.asarray(im_rgbd.color)
    kernel = np.ones((1,1),np.uint8)
    kernel2 = np.ones((15,15),np.uint8)


    while True:
        if True:
            im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
            img=np.asarray(im_rgbd.color)
            depth=np.asarray(im_rgbd.depth)

            ori=img.copy()
            cv2.imshow("image", img)
            cv2.waitKey(1)
            i=0
            #msvcrt.kbhit() and msvcrt.getwche() == 's'
        try:
            if msvcrt.kbhit() and msvcrt.getwche() == 's':
                
                #blurred = cv2.GaussianBlur(img[i], (17, 17), 0)
                img1= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img1=cv2.GaussianBlur(img1,(13,13),0)

                img_t=cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

                img_t=cv2.morphologyEx(img_t,cv2.MORPH_OPEN,kernel)
                img_t=cv2.morphologyEx(img_t,cv2.MORPH_CLOSE,kernel2)
                #img_t=cv2.morphologyEx(img_t,cv2.MORPH_DILATE,kernel2)

                contours, hierarchy = cv2.findContours(img_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                            

                while True: 
                    BOX=[]
                    img2=img.copy()

                    for l in range(len(contours)):
                        cnt = contours[l]
                        if cv2.contourArea(cnt)>1000 and cv2.arcLength(cnt,True)<2400:
                            
                            #cv2.drawContours(img[i], contours,l, (0,255,0),thickness=cv2.FILLED)
                            rect = cv2.minAreaRect(cnt)
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            #box = cv2.convexHull(cnt)

                            BOX.append(box)
                
                            #cv2.drawContours(img2,[box],0,(0,0,255),2)
                            #print("cont")
                            #cv2.imshow("image", img2)   
                            #cv2.waitKey(0)
                    i=0
                    print("boxsize",len(BOX))
                    while True:
                        if i==len(BOX):
                            i=0

                        if msvcrt.getwche() == 'p':
                            print("STOP")
                            raise StopIteration

                        if  msvcrt.getwche() == 'w':
                            cv2.destroyAllWindows()

                            while(True):
                                #venstre op=BOX[i][0]
                                #højre op=BOX[i][1]
                                #venstre ned=BOX[i][2]
                                #højre ned=BOX[i][3]
                                stencil = np.zeros(((img.shape))).astype(img.dtype)

                                color = [255, 255, 255]
#                                    points=numpy.array([BOX[i][0], BOX[i][1],BOX[i][1],BOX[i][1]])
                                cv2.fillPoly(stencil, np.int32(([BOX[i]])), color)


                                depth=cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)
                                stencil2 = np.zeros(((depth.shape))).astype(depth.dtype)

                                result_img = cv2.bitwise_and(ori, stencil)
                                print(np.shape(depth),np.shape(stencil))
                                result_depth=cv2.bitwise_and(depth, stencil2)
                                result_depth=cv2.cvtColor(depth,cv2.COLOR_RGB2GRAY)
                                return result_img, result_depth

                        for k in range(len(BOX)):
                            if k!=i:
                                cv2.drawContours(img2,[BOX[k]],0,(0,0,255),2) 

                        cv2.drawContours(img2,[BOX[i]],0,(255,0,0),2)
                        cv2.imshow("image", img2)   
                        cv2.waitKey(1)

                        if  msvcrt.getwche() == 'q':
                            i=i+1
        except StopIteration: pass

color,depth=run()


color=o3d.geometry.Image((color))
depth=o3d.geometry.Image((depth))

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,convert_rgb_to_intensity=False)
print("intrinc",type(intrinsic))
print("rgb",type(rgbd))
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,intrinsic)
#pcd = pcd.voxel_down_sample(voxel_size=0.0)
#voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
print(np.asarray(pcd.colors))
print(np.shape(np.sum(np.asarray(pcd.colors),axis=0) ))
sel=np.where(np.sum(np.asarray(pcd.colors),axis=1) > 0)[0]
pcd=pcd2 = pcd.select_by_index(sel)




pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*3, max_nn=400))
fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*7, max_nn=900))
L=CenterOfPCD(np.asarray(pcd.points))
fph = np.array(np.asarray(fph.data)).T
#fph=np.append(fph,L,axis=1)

aff=reg.predict(fph)
aff=np.asarray(aff).reshape((len(pcd.points),1))
nps=np.zeros(np.shape(aff))

#pcd.colors=o3d.utility.Vector3dVector(np.concatenate((aff,np.asarray(np.asarray(pcd.colors)[:, :1]),np.asarray(np.asarray(pcd.colors)[:, 1:2])),axis=1))
pcd.colors=o3d.utility.Vector3dVector(np.concatenate((aff,nps,nps),axis=1))

o3d.visualization.draw_geometries([pcd])

np.save('pcd.npy',np.asanyarray(pcd.points))
np.save('pcd_c.npy',pcd.colors)


#o3d.visualization.draw_geometries([pcd])


