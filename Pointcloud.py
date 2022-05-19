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
import sys
import urx
import time
import math
import math3d as m3d
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

rob = urx.Robot("172.31.1.115")
print("pose:",rob.get_pose())
a = 0.4
v = 0.5
startingJoint=[0.4976164996623993, -1.8964617888080042, 1.7800350189208984, 0.837715744972229, 1.5187908411026, 1.4695764780044556]
rob.movej(startingJoint,a,v)
#rob.back(-0.2, a, v)


pickle_file = "C:\\data_for_learning\\DebugState.pickle" ### Write path for the full_shape_val_data.pkl file ###
print("Pickle learner")
#pickle_file = "C:\\data_for_learning\\RegressionGrasp2.pickle" ### Write path for the full_shape_val_data.pkl file ###       
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)
print("pickle Learned")


with open("CameraSetup.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0)

rs.start_capture(True)  # true: start recording with capture  
mtx=np.load("RealSenseCameraParams\\RealSenseCameraParmas1280x720.npy")
intrinsic=o3d.camera.PinholeCameraIntrinsic(640,480 ,mtx[0][0],mtx[1][1],mtx[0][2],mtx[1][2])
o3d.t.io.RealSenseSensor.list_devices()


im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the

def CalcCam2ToolMatrix():
    Cam2Base = np.array([[-1, 0, 0,-0.750],
                         [ 0, 1, 0,-0.750],
                         [ 0, 0,-1, 0.131],
                         [ 0, 0, 0, 1.000]])
    Gripper2Base = np.array([[-0.99614576,  0.0281356 , -0.08307836, -0.69665],
                             [ 0.02663937,  0.99946331,  0.01906395, -0.71217],
                             [ 0.08357015,  0.01677732, -0.99636065,  0.00000],
                             [ 0,           0,           0,           1      ]])
    gripper2cam0 = np.dot(np.linalg.inv(Cam2Base),     Gripper2Base)
    gripper2cam1 = np.dot(np.linalg.inv(Gripper2Base), Cam2Base    ) 
    gripper2cam2 = np.dot(Gripper2Base, np.linalg.inv(Cam2Base)    ) 
    gripper2cam3 = np.dot(Cam2Base,     np.linalg.inv(Gripper2Base))
    #print(gripper2cam0)
    #print(gripper2cam1)
    #print(gripper2cam2)
    #print(gripper2cam3)
    return gripper2cam0, gripper2cam1, gripper2cam2, gripper2cam3


def HomogenousTransformation(pose):
    trans=pose.pos
    rot=pose.orient

    print(rot)

    print(rob.get_pose())
    

    print(np.shape( rob.get_orientation()))
    translation=np.asarray([trans[0], trans[1], trans[2]])
    Transformation = np.eye(4)
    r=R.from_rotvec([rot[0],rot[1],rot[2]])
    rotation=r.as_matrix()
    print(type(R.from_rotvec(rotation)))
    Transformation[:3, :3] = np.asanyarray(R.from_rotvec(rotation))
    Transformation[0, 3]=translation[0]
    Transformation[1, 3]=translation[1]
    Transformation[2, 3]=translation[2]

    return Transformation


def RemoveBackGround(rgbd,d):
    a=np.asarray(im_rgbd.depth)

    a=(a < d*1000)*a

    im=np.asarray(im_rgbd.color)

    Im_Rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(im), o3d.geometry.Image(a),convert_rgb_to_intensity=False)

    return Im_Rgbd


def Rotation(rot,iteration,resulution):
    Circle=np.linspace(0, np.pi, 4,endpoint=True)
    
    angle=rot+Circle[iteration]
    
    if angle>2*np.pi:  
        angle=angle-2*np.pi
        return angle
    if angle<0:
        angle=angle+2*np.pi
        return angle
    else:
        return angle


def CenterOfPCD(PCD):
    m_xyz=np.mean(PCD,axis=0)
    L=np.linalg.norm(m_xyz-PCD,axis=1).reshape((len(PCD),1))
    L=(L-min(L))/(max(L)-min(L))
    return L


def rotationMatrixToEulerAngles(R) :
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(theta) :
    #theta=(theta)[0]
    print(np.shape(theta))
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])

    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])

    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R


def grasp_Positions(pcd,aff):

    pcd=ColorAffordance(aff,pcd,[0,0,1])
    k=80
    position=np.expand_dims(np.squeeze(np.asarray(aff[aff.argsort(axis=0)[-k:]])),axis=1)      # these are sorted off
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
    #for i in range(len(pcd_.points)):
    for i in range(1):
        [k, idx, _] = pcd_tree.search_radius_vector_3d((pcd_.points)[i], 0.24/4.2)
        pcd2 = pcd.select_by_index(idx)

        np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        o3d.visualization.draw_geometries([pcd])

        cov_m=np.cov(np.asarray(pcd2.points).transpose())
        cov_m[np.isnan(cov_m)] = 0
        S=np.linalg.svd(cov_m,False,False)  #no full svd nor UV
        k1k2=([S[0],S[1]])
        m_curv=([(S[0]+S[1]/2)])
        g_curv=([(S[0]*S[1]/2)])
        #T[0, 3]=(np.asarray(pcd_.points)[i,0])
        #T[1, 3]=(np.asarray(pcd_.points)[i,1])
        #T[2, 3]=(np.asarray(pcd_.points)[i,2])
        mean=np.mean(np.asarray(pcd_.points),axis=0)
        T[0, 3]=mean[0]
        T[1, 3]=mean[1]
        T[2, 3]=mean[2]
        

        print("point",T[0, 3],T[1, 3],T[2, 3])
        R=rotationMatrixToEulerAngles(PfLEO_OBB.R)
        for r in range(1):
            for p in range(1):
                for y in range(1):
                    Rot_xyz=(np.array([Rotation(R[0],r,3),Rotation(R[1],p,3),Rotation(R[2],y,3)]))
                    #Rot_xyz=(np.array([R[0],R[1],R[2]]))

                    print("rot: ",Rot_xyz)
                    T[:3, :3]=eulerAnglesToRotationMatrix(Rot_xyz)
                    x=(([T[0, 3]/np.median(pcd_.points[0],axis=0),T[1, 3]/np.median(pcd_.points[0],axis=0),T[2, 3]/np.median(pcd_.points[0],axis=0),R[0]-Rot_xyz[0],R[1]-Rot_xyz[1],R[2]-Rot_xyz[2],(pcd_.colors)[i],
                    k1k2,m_curv,g_curv]))
                    
                    if len(X)==0:
                        X=np.transpose(np.expand_dims(x,axis=1))
                        Poses=[T]
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


def ControlGripper(command = 'open'):
    if (command == 'open'):
        scriptLocation ='gripperScripts//openg.script'
        print("open gripper")
    elif (command == 'close'):
        scriptLocation ='gripperScripts//closeg.script'
        print("closing gripper")
    else:
        print("Error, unknown command, either input 'open' or 'close'")
        return
    with open(scriptLocation, 'r') as script:
        rob.send_program(script.read())
        time.sleep(4)


def SVD_Principal_Curvature(Pointcloud,radius,factor):
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


def Extract_Feature(pcd,factor):
    #Non Normalized distance: 
    L=CenterOfPCD(np.asarray(pcd.points))
#curvature features:
    cur=SVD_Principal_Curvature(pcd,0.24,factor)
    x=np.append(cur,L,axis=1)
# Fast Feature
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.1*factor))
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamRadius(radius=0.2*factor))
    
    fph = np.array(np.asarray(fph.data)).T
    fph=np.append(fph,L,axis=1)
    
    x=np.append(cur,fph,axis=1)

    return x


def ColorAffordance(aff,pcd,color):
    np_colors=np.zeros((len(pcd.points),2))
    np_colors=(np.concatenate((aff,np_colors),axis=1))
    #maxaff=np.argmax(aff)
    #np_colors[maxaff][:]=[0,0,1]

    pcd.colors=o3d.utility.Vector3dVector(np_colors)

    return pcd


Method_Tsai=1
Method_Andref=2
Method_Dadnilist=3
Method_Horud=4
Method_Park=5


def calculateT2B(world2camMat,meth):
    if meth==1:
        cam2gripperMat = np.load("Calibration__Data\\HandEyeTransformation.npy")
    if meth==2: 
        cam2gripperMat = np.load("Calibration__Data\\HandEyeTransformation_andereff.npy")
    if meth==3: 
        cam2gripperMat = np.load("Calibration__Data\\HandEyeTransformation_DANIILIDIS.npy")
    if meth==4: 
        cam2gripperMat = np.load("Calibration__Data\\HandEyeTransformation_HORAUD.npy")
    if meth==5:
        cam2gripperMat = np.load("Calibration__Data\\HandEyeTransformation_park.npy")

    meth=1
    gripper2baseMat = rob.get_pose()

    gripper2baseMat = gripper2baseMat.get_matrix()
    cam2gripperMat0, cam2gripperMat1, cam2gripperMat2, cam2gripperMat3 = CalcCam2ToolMatrix()
    #world2base = (gripper2baseMat) * (cam2gripperMat) * (world2camMat)

    world2base0 = np.dot(gripper2baseMat, np.dot(cam2gripperMat0, world2camMat))
    world2base1 = np.dot(gripper2baseMat, np.dot(cam2gripperMat1, world2camMat))
    world2base2 = np.dot(gripper2baseMat, np.dot(cam2gripperMat2, world2camMat))
    world2base3 = np.dot(gripper2baseMat, np.dot(cam2gripperMat3, world2camMat))
    if (meth == 0):
        return world2base0
    if (meth == 1):
        return world2base1
    if (meth == 2):
        return world2base2
    if (meth == 3):
        return world2base3



def Clustering(pcl,eps,min_points):
    labels = np.array(
        pcd.cluster_dbscan(eps=0.02, min_points=30, print_progress=True))
    max_label = labels.max()
    Clus=[]
    for i in range(max_label):
        id=np.where(labels==i)[0]
        pcl_i=pcl.select_by_index(id)
        Clus.append(pcl_i)

    return Clus

time_b=time.time()
im_rgbd=RemoveBackGround(im_rgbd,1)

for i in np.asanyarray(im_rgbd.depth):
    #print(i)
    pass

print("time to remove background ",time.time()-time_b )
ori=o3d.geometry.PointCloud.create_from_rgbd_image(im_rgbd,intrinsic)
pcd = ori.voxel_down_sample(voxel_size=0.007)



pcd_tree = o3d.geometry.KDTreeFlann(pcd)

[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[5130], 0.24/4)
pcd_ = pcd.select_by_index(idx)
pcd.paint_uniform_color([0,0,0])
np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

o3d.visualization.draw_geometries([pcd  ])

time_b=time.time()
plane_model, inliers = pcd.segment_plane(distance_threshold=0.03,
                                         ransac_n=400,
                                         num_iterations=10000)

                                        
print("time to remove Segment ",time.time()-time_b )

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
pcd = pcd.select_by_index(inliers, invert=True)


#o3d.visualization.draw_geometries([pcd ,inlier_cloud ])
time_b=time.time()

#o3d.visualization.draw_geometries([pcd])

Clus = Clustering(pcd,0.02,100)

def pointPCD(Cluster):
    i=0
    while(True):
        for j in range(len(Cluster)):
            Cluster[j].paint_uniform_color([0, 0, 0])
            Cluster[i].paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries(Cluster)
        if msvcrt.kbhit() and msvcrt.getwche() == 's':
            return Cluster[i]
        else:
            i=i+1             
def normalizeCluster(Cluster):
    minX = min(Cluster[:,0])
    print("minimum x value is:", minX)
    minY = min(Cluster[:,1])
    print("minimum y value is:", minY)
    minZ = min(Cluster[:,2])
    print("minimum z value is:", minZ)

    for currentPoint in range(len(Cluster)):
        Cluster[currentPoint,0] -= minX
        Cluster[currentPoint,1] -= minY
        Cluster[currentPoint,2] -= minZ

    maxVal = findMax(Cluster)
    for currentPoint in range(len(Cluster)):
        Cluster[currentPoint,0] = Cluster[currentPoint,0]/maxVal
        Cluster[currentPoint,1] = Cluster[currentPoint,1]/maxVal
        Cluster[currentPoint,2] = Cluster[currentPoint,2]/maxVal
    return Cluster


print(len(Clus))
if (Clus):
    pcd=pointPCD(Clus)
    pcd.paint_uniform_color([1, 0, 0])  

print("Visualizing full point cloud:")
o3d.visualization.draw_geometries([pcd,ori])

mean=np.mean(np.asarray(pcd.points),axis=0)

print(mean)
print(np.linalg.norm(mean))
x=Extract_Feature(pcd,1/4)
aff=reg.predict(x)
aff=np.asarray(aff).reshape((len(pcd.points),1))
Poses,X,pc=grasp_Positions(pcd,aff)

for i in range(1):    
    rob.movej(startingJoint,a,v)
    print("Attempting new method")
    B=calculateT2B((Poses[0]),i)
    print(B)
    print(rotationMatrixToEulerAngles(B[:3, :3]))
    #ControlGripper(command = 'close')
    mytcp = m3d.Transform()
    mytcp.pos.x = B[0,3]
    mytcp.pos.y = B[1,3]
    if (B[2,3] < 0.05):
        mytcp.pos.z = 0.05
    else:
        mytcp.pos.z = B[2,3]

    mytcp.orient.rotate_yb(-math.pi/2)
    mytcp.orient.rotate_zb(math.pi/4)

    prePos = mytcp
    #prePos.pos.x += 0.10
    #prePos.pos.y += 0.10
    try:
        rob.set_pose(prePos,a,v,command = 'movej')
    except:
        print("could not set pose")

    try:
        rob.set_pose(mytcp,a,v,command = 'movel')
    except:
        print("could not set pose")
    #ControlGripper(command = 'open')
    #rob.back(-0.1, a, v)
    #ControlGripper(command = 'close')
    time.sleep(2)

rob.stop()
rob.close()
sys.exit()
 
if False:
    x=Extract_Feature(pcd,1/4)
    aff=reg.predict(x)
    aff=np.asarray(aff).reshape((len(pcd.points),1))
    pcd=ColorAffordance(aff,pcd,1)
    o3d.visualization.draw_geometries([pcd])