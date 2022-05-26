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
GainAffordanceBytenLargestValues=1
GainAffordanceCummualativt=2


import matplotlib.pyplot as plt

rob = urx.Robot("172.31.1.115")
print("pose:",rob.get_pose())
a = 0.4
v = 0.5
startingJoint=[0.33082714676856995, -2.0115001837359827, 1.706066608428955, 1.1847649812698364, 1.4761427640914917, 1.2847598791122437]
rob.movej(startingJoint,a,v)
rob.back(-0.3, a, v)


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
mtx=mtx=np.load("RealSenseCameraParams\\RealSenseCameraParmas1280x720.npy")
intrinsic=o3d.camera.PinholeCameraIntrinsic(1280,720 ,mtx[0][0],mtx[1][1],mtx[0][2],mtx[1][2])
o3d.t.io.RealSenseSensor.list_devices()


im_rgbd = rs.capture_frame(True, True)  # wait for frames and align the

def CalcCam2ToolMatrix():
    Cam2Base = np.array([[ 0, 1, 0,-0.50],
                         [ 1, 0, 0,-0.50],
                         [ 0, 0,-1, 0.131],
                         [ 0, 0, 0, 1.000]])
    Gripper2Base = np.array([[-0.99988494, -0.01234059,  0.00882146, -0.42776],
                             [-0.01228032,  0.9999011 ,  0.00685469, -0.46564],
                             [-0.00890518,  0.00674557, -0.9999376,   0      ],
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

def MaxApffordance(pcd,aff,method):
    
    if method==1:
        k=10
        position=np.expand_dims(np.squeeze(np.asarray(aff[aff.argsort(axis=0)[-k:]])),axis=1)
        return position      # these are sorted off
    if method==2:

        np_colors=np.zeros((len(pcd.points),1))

        np_colors=(np.concatenate((aff,np_colors,np_colors),axis=1))

        pcd.colors=o3d.utility.Vector3dVector(np_colors)
        #o3d.visualization.draw_geometries([pcd])

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        #o3d.visualization.draw_geometries([pcd])
        index=[]
        points=[]
        for i in range(len(np.array(pcd.points))):
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], 0.17)
            pcd2 = pcd.select_by_index(idx)
            #o3d.visualization.draw_geometries([pcd2])
            index.append(np.sum(np.array(pcd2.colors)))
            points.append(pcd.points[i])
            #print(np.array(pcd2.colors))
            #print(np.sum(np.array(pcd2.colors)))
                
   

        index=np.asarray(index)
        points=np.array(points)
        k=17
        print(np.shape(index))
       # position=np.squeeze((index),axis=1)
        print(np.shape(index))
        position=(-index).argsort()[:k]
        xyz=points[position]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector((xyz*1.001))
        #o3d.visualization.draw_geometries([pcd2])


        return pcd2



        return list((position))


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

def grasp_Positions2(pcd,aff):
    pcd.paint_uniform_color([0, 0, 0])
    pcd_=MaxApffordance(pcd,aff,GainAffordanceCummualativt)     # these are sorted off
    
    print(np.shape(np.asanyarray(pcd_.points)))
    pcd_.paint_uniform_color([0, 1, 0])
    pcd.paint_uniform_color([0, 0, 0])

    #o3d.visualization.draw_geometries([pcd_,pcd,pcd_])
    return pcd_

    
def ColorAffordance(aff,pcd,color):
    pcd2=pcd
    np_colors=np.zeros((len(pcd.points),2))
    np_colors=(np.concatenate((aff,np_colors),axis=1))
    pcd2.colors=o3d.utility.Vector3dVector(np_colors)
   # o3d.visualization.draw_geometries([pcd2])

    np_colors=np.zeros((len(pcd.points),2))
    aff_=np.asarray(aff).reshape((len(pcd.points),1))
    aff_[aff_<=0.1313572330635844]=0
    aff_[aff_>0.1313572330635844]=1

    np_colors=(np.concatenate((aff_,np_colors),axis=1))
    #maxaff=np.argmax(aff)
    #np_colors[maxaff][:]=[0,0,1]
    pcd.paint_uniform_color([0, 0, 0])

    pcd.colors=o3d.utility.Vector3dVector(np_colors)
    #o3d.visualization.draw_geometries([pcd])

    pcd3 = pcd.voxel_down_sample(voxel_size=0.07)
    print("numper of samples for pointcloud", np.shape(np.asarray(pcd3.points)))
    #o3d.visualization.draw_geometries([pcd3])

    return pcd

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


def SVD_Principal_Curvature(points,radius,factor):
    Pointcloud= o3d.geometry.PointCloud()
    Pointcloud.points = o3d.utility.Vector3dVector((points))
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
    L=CenterOfPCD(np.asarray(pcd))
#curvature features:
    cur=SVD_Principal_Curvature(pcd,0.14,factor)
    x=np.append(cur,L,axis=1)
# Fast Feature
    Pointcloud= o3d.geometry.PointCloud()
    Pointcloud.points = o3d.utility.Vector3dVector((pcd))
    Pointcloud.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.1*factor))
    fph=o3d.pipelines.registration.compute_fpfh_feature(Pointcloud, o3d.geometry.KDTreeSearchParamRadius(radius=0.2*factor))
    
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
        cam2gripperMat0 = np.load("Calibration__Data\\HandEyeTransformation.npy")
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
plane_model, inliers = pcd.segment_plane(distance_threshold=0.011,
                                         ransac_n=900,
                                         num_iterations=1000)

                                        
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
class normalizeCluster:
    def __init__(self, Cluster):
        self.minX = min(Cluster[:,0])
        self.minY = min(Cluster[:,1])
        self.minZ = min(Cluster[:,2])

        for currentPoint in range(len(Cluster)):
            Cluster[currentPoint,0] -= self.minX
            Cluster[currentPoint,1] -= self.minY
            Cluster[currentPoint,2] -= self.minZ

        self.maxVal = np.amax(Cluster)
        for currentPoint in range(len(Cluster)):
            Cluster[currentPoint,0] = Cluster[currentPoint,0]/self.maxVal
            Cluster[currentPoint,1] = Cluster[currentPoint,1]/self.maxVal
            Cluster[currentPoint,2] = Cluster[currentPoint,2]/self.maxVal
        self.Cluster = Cluster

    def unnormalize(self, Cluster):
        for currentPoint in range(len(Cluster)):
            Cluster[currentPoint,0] = Cluster[currentPoint,0]*self.maxVal
            Cluster[currentPoint,1] = Cluster[currentPoint,1]*self.maxVal
            Cluster[currentPoint,2] = Cluster[currentPoint,2]*self.maxVal

        for currentPoint in range(len(Cluster)):
            Cluster[currentPoint,0] += self.minX
            Cluster[currentPoint,1] += self.minY
            Cluster[currentPoint,2] += self.minZ
        return Cluster


print(len(Clus))
if (Clus):
    pcd=pointPCD(Clus)
    pcd.paint_uniform_color([1, 0, 0])  

print("Visualizing full point cloud:")
o3d.visualization.draw_geometries([pcd,ori])
ori = o3d.geometry.PointCloud()
ori.points = o3d.utility.Vector3dVector(np.asanyarray(pcd.points))
getNorm=normalizeCluster(np.asanyarray(pcd.points))
xyz = getNorm.Cluster
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector((xyz))

x=Extract_Feature(np.asarray(pcd.points),1)
aff=reg.predict(x)
aff=np.asarray(aff).reshape((len(pcd.points),1))
aff_=aff

pcd=ColorAffordance(aff,pcd,1)
points=grasp_Positions2(pcd,aff_)

points=getNorm.unnormalize(np.asanyarray(points.points))
pcd_ = o3d.geometry.PointCloud()
pcd_.points = o3d.utility.Vector3dVector((points))
pcd_.paint_uniform_color([0, 1, 0])
ori.paint_uniform_color([0, 0, 0])

o3d.visualization.draw_geometries([pcd_,ori])

print(np.asanyarray(pcd_.points)[0], "norm", np.linalg.norm(np.asanyarray(pcd_.points)[0]))
print(np.shape(pcd_.points[0]))
B=np.eye(4)
print(B)
points=np.asanyarray(pcd_.points)[0]
print(np.shape(points))
B[0,3]=points[0]
B[1,3]=points[1]
B[2,3]=points[2]

PfLEO_OBB = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_.points)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
T = np.eye(4)
T[:3, :3]= PfLEO_OBB.R
T[0, 3]=points[0]
T[1, 3]=points[1]
T[2, 3]=points[2]

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
import copy
mesh_t = copy.deepcopy(mesh).transform(T)
mesh_t.scale(0.1, center=mesh_t.get_center())

o3d.visualization.draw_geometries([ori,pcd_,mesh_t])


for i in range(1):    
    #rob.movej(startingJoint,a,v)
    print("Attempting new method")
    W2B=calculateT2B(T,1)
    print(W2B)
    print(rotationMatrixToEulerAngles(W2B[:3, :3]))
    #ControlGripper(command = 'close')
    mytcp = m3d.Transform()
    mytcp.pos.x = W2B[0,3]
    mytcp.pos.y = W2B[1,3]
    if (W2B[2,3] < 0):
        mytcp.pos.z = 0
        W2B[2,3]=0
    else:
        mytcp.pos.z = W2B[2,3]
    rob.set_pose(mytcp,a,v,command = 'movel')

    print(np.shape(W2B[:3, :3]),type(W2B[:3, :3]))
    mytcp.orient.rotate_xb(math.pi/2)
    mytcp.orient
    prePos = m3d.Transform()
    prePos.orient = mytcp.orient
    prePos.pos.x = mytcp.pos.x + 10
    prePos.pos.y = mytcp.pos.y + 10


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

print(W2B)


np.eye()

if False:
    for i in range(1):    
        rob.movej(startingJoint,a,v)
        print("Attempting new method")
        B=calculateT2B((Poses[0]),i+1)
        print(B)
        print(rotationMatrixToEulerAngles(B[:3, :3]))
        #ControlGripper(command = 'close')
        mytcp = m3d.Transform()
        mytcp.pos.x = B[0,3]
        mytcp.pos.y = B[1,3]
        if (B[2,3] < 0):
            mytcp.pos.z = 0
        else:
            mytcp.pos.z = B[2,3]

        mytcp.orient.rotate_xb(math.pi/2)

        prePos = mytcp
        prePos.pos.x += 10
        prePos.pos.y += 10
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