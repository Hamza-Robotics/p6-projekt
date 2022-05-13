#from contextlib import closing
#import p6
import json
from unittest.mock import sentinel
from cv2 import cvtColor
import pyrealsense2 as rs
#import time
import open3d as o3d
import numpy as np
import cv2
import msvcrt
import pickle
import sys
import urx
import time
import math
import math3d as m3d

rob = urx.Robot("172.31.1.115")
print("pose:",rob.get_pose())

o3d.t.io.RealSenseSensor.list_devices()


pickle_file = "C:\\data_for_learning\\RegressionGrasp2.pickle" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)
    #reg=2

pipeline = rs.pipeline()

cfg = pipeline.start() # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

#intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[231.7225,0,161.8237],[0,229.9110,80.4734],[0,0,1.0000]])))
#intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])))
#intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,intr.fx,intr.fy,intr.ppx,intr.ppy)

#intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,317.9628,320.223,216.6219,114.8350)

def Rotation(rot,iteration,resulution):
    Circle=np.linspace(0, 2*np.pi, resulution)
    
    angle=rot-Circle[0]
    
    #if angle>2*np.pi:  
      #  angle=angle-2*np.pi
     #   return angle
    #if angle<0:
     #   angle=angle+2*np.pi
    #    return angle
   # else:
    return angle

def CenterOfPCD(PCD):
    m_xyz=np.mean(PCD,axis=0)
    L=np.linalg.norm(m_xyz-PCD,axis=1).reshape((len(PCD),1))
    L=(L-min(L))/(max(L)-min(L))
    return L

def grasp_Positions(pcd,aff):

    pcd=ColorAffordance(aff,pcd,[0,0,1])
    k=80
    position=np.expand_dims(np.squeeze(np.asarray(aff[aff.argsort(axis=0)[-k:]])),axis=1)      
    #np.asarray(pcd.colors)[aff.argsort(axis=0)[-k:], :] = [0, 1, 1]
    #o3d.visualization.draw_geometries([pcd])
    try:
        pass

          
    except:
        print("no affordance")
        return np.asarray([0]),np.asarray([0]),np.asarray([0])
    PfLEO_OBB = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T = np.eye(4)
    T[:3, :3]= PfLEO_OBB.R
    X=np.asarray([])
    Poses=np.asarray([])

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    pcd_ = pcd.select_by_index(position)
    pcd_.paint_uniform_color([0,0,0])
    for i in range(len(pcd_.points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d((pcd_.points)[i], 0.007*factor)
        pcd2 = pcd.select_by_index(idx)

        #np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        #o3d.visualization.draw_geometries([pcd])

        cov_m=np.cov(np.asarray(pcd2.points).transpose())
        cov_m[np.isnan(cov_m)] = 0
        S=np.linalg.svd(cov_m,False,False)  #no full svd nor UV
        k1k2=([S[0],S[1]])
        m_curv=([(S[0]+S[1]/2)])
        g_curv=([(S[0]*S[1]/2)])
        T[0, 3]=np.asarray(pcd_.points)[i][0]
        T[1, 3]=np.asarray(pcd_.points)[i][1]
        T[2, 3]=np.asarray(pcd_.points)[i][2]
        R,_=cv2.Rodrigues(PfLEO_OBB.R)

        for r in range(1):
            for p in range(1):
                for y in range(1):
                    Rot_xyz=(np.array([[Rotation(R[0],r,3),Rotation(R[1],p,3),Rotation(R[2],y,3)]]))
                    print("rot: ",Rot_xyz)
                    T[:3, :3],_=cv2.Rodrigues(Rot_xyz)
                    rpy,_=cv2.Rodrigues(T[:3, :3])
                    print( T[:3, :3])
                    x=np.concatenate(([T[0, 3]/np.median(pcd_.points[0],axis=0),T[1, 3]/np.median(pcd_.points[0],axis=0),T[2, 3]/np.median(pcd_.points[0],axis=0)],R[0]-rpy[0],R[1]-rpy[1],R[2]-rpy[2],(pcd_.colors)[i],
                    k1k2,m_curv,g_curv))
                    
                    if len(X)==0:
                        X=np.transpose(np.expand_dims(x,axis=1))
                        Poses=[T]
                        print("100", np.shape(np.array(Poses)))
                    else:
                        X=np.concatenate((X,np.transpose(np.expand_dims(x,axis=1))),axis=0)
                        Poses=np.concatenate((Poses,[T]),axis=0)

            #X=np.concatenate( (np.concatenate((X,np.transpose(np.expand_dims(x,axis=1))),axis=1),x_e),axis=0)

    return np.asarray(Poses), np.asarray(X), pcd_

def ColorAffordance(aff,pcd,color):
    np_colors=np.zeros((len(pcd.points),2))
    np_colors=(np.concatenate((aff,np_colors),axis=1))
    #maxaff=np.argmax(aff)
    #np_colors[maxaff][:]=[0,0,1]

    pcd.colors=o3d.utility.Vector3dVector(np_colors)

    return pcd

factor=10

def SVD_Principal_Curvature(Pointcloud,radius):
    k1k2=[]
    m_curv=[]
    g_curv=[]
    pcd_tree = o3d.geometry.KDTreeFlann(Pointcloud)

    for i in range(len(Pointcloud.points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d((Pointcloud.points)[i], radius*factor)
        pcd_ = pcd.select_by_index(idx)
        Pointcloud.paint_uniform_color([0,0,0])

        #np.asarray(Pointcloud.colors)[idx[1:], :] = [0, 0, 1]
        #o3d.visualization.draw_geometries([Pointcloud])
        cov_m=np.cov(np.asarray(pcd_.points).transpose())
        cov_m[np.isnan(cov_m)] = 0
        S=np.linalg.svd(cov_m,False,False)  #no full svd nor UV
        k1k2.append([S[0],S[1]])
        m_curv.append([(S[0]+S[1]/2)])
        g_curv.append([(S[0]*S[1]/2)])

    return np.concatenate((np.asarray(k1k2), np.asarray(m_curv), np.asarray(g_curv)),axis=1)

def Extract_Feature(pcd):
    #Non Normalized distance: 
    L=CenterOfPCD(np.asarray(pcd.points))
#curvature features:
    cur=SVD_Principal_Curvature(pcd,0.007)
    x=np.append(cur,L,axis=1)
# Fast Feature
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.1*factor))
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamRadius(radius=0.06*factor))
    
    fph = np.array(np.asarray(fph.data)).T
    fph=np.append(fph,L,axis=1)
    
    x=np.append(cur,fph,axis=1)

    return x

def run():


    with open("CameraSetup.json") as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

    rs = o3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0)
    rs.start_capture(True)  # true: start recording with capture  

    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the

    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
    img=np.asarray(im_rgbd.color)
    print(    np.shape(img))
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


                                stencil2 = np.zeros(((depth.shape))).astype(depth.dtype)
                                result_img = cv2.bitwise_and(ori, stencil)
                                #cv2.imshow("img",result_img)
                                print(np.shape(depth.reshape(720, 1280)),np.shape(result_img))
                                #cv2.waitKey(0)
                                foo_=(cv2.cvtColor(result_img,cv2.COLOR_RGB2GRAY))
                                print(type(foo_), np.shape(foo_))
                                for iy, ix in np.ndindex(np.shape(foo_)):
                                    if foo_[iy,ix]==0:
                                        depth[iy,ix]=0
                              
                                                                #depth=cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)

                                #result_depth=cv2.bitwise_and(depth, stencil2)
                                #cv2.imshow("img2",result_depth*100)
                                #cv2.waitKey(0)

                                #result_depth=cv2.cvtColor(depth,cv2.COLOR_RGB2GRAY)

                                return result_img, depth
                                

                        for k in range(len(BOX)):
                            if k!=i:
                                cv2.drawContours(img2,[BOX[k]],0,(0,0,255),2) 

                        cv2.drawContours(img2,[BOX[i]],0,(255,0,0),2)
                        cv2.imshow("image", img2)   
                        cv2.waitKey(1)

                        if  msvcrt.getwche() == 'q':
                            i=i+1
        except StopIteration: pass

def calculateT2B(world2camMat, inverse = False):
    cam2gripperMat = np.load('CameraCalib.npy')
    base2gripperMat = rob.get_pose()
    gripper2baseMat = base2gripperMat.get_inverse()
    gripper2baseMat = gripper2baseMat.get_matrix()
    if inverse == False:
        world2base = world2camMat * cam2gripperMat * gripper2baseMat
    else:
        world2base = gripper2baseMat * cam2gripperMat * world2camMat
    return world2base


while True:
    color,depth=run()
    print("here")
    color=o3d.geometry.Image((color))
    depth=o3d.geometry.Image((depth))
    intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,317.9628,320.223,216.6219,114.8350)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,intrinsic)
    pcd=pcd.voxel_down_sample(voxel_size=0.02)
    o3d.visualization.draw_geometries([pcd])

    x=Extract_Feature(pcd)

    aff=reg.predict(x)
    aff=np.asarray(aff).reshape((len(pcd.points),1))
    pcd.paint_uniform_color([0, 0, 0])

    T,X,pcd_=grasp_Positions(pcd,aff)
    print("    qqqq   ")

    #print(cam2gripperMat)
    print("    ssss   ")

    print(T[0])

    print("    ddd   ")
    print(calculateT2B(T[0]))
    print(calculateT2B(T[0], inverse = True))

    #pcd.colors=o3d.utility.Vector3dVector(np.concatenate((aff,np.asarray(np.asarray(pcd.colors)[:, :1]),np.asarray(np.asarray(pcd.colors)[:, 1:2])),axis=1))

    o3d.visualization.draw_geometries([pcd])

    np.save('pcd2.npy',np.asanyarray(pcd.points))
    np.save('pcd2_c.npy',pcd.colors)



    #o3d.visualization.draw_geometries([pcd])


