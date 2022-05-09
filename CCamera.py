from ctypes.wintypes import PLCID
import json
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import msvcrt
import pyrealsense2 as rs

def CenterOfPCD(PCD):
    m_xyz=np.mean(PCD,axis=0)
    L=np.linalg.norm(m_xyz-PCD,axis=1).reshape((len(PCD),1))
    L=(L-min(L))/(max(L)-min(L))
    return L

def Segmentation(pcl):
    plane_model, inliers = pcl.segment_plane(distance_threshold=0.20,
                                        ransac_n=3,
                                        num_iterations=2000)       
    inlier_cloud = pcl.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcl.select_by_index(inliers, invert=True)
    return inlier_cloud,outlier_cloud
def Clustering(pcl,eps,min_points):
    labels = np.array(pcl.cluster_dbscan(eps, min_points,True))
    #dbscan labeller alle point clouds. Man finder den point cloud med højeste label. En cluster er alle de points med samme label. 
    max_label = labels.max()
    Clus=[]


    for i in range(max_label+1):
        id=np.where(labels==i)[0]
        pcl_i=pcl.select_by_index(id)
        Clus.append(pcl_i)

        return Clus

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

im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
       
pcd_o = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd,intrinsic)
pcd=pcd_o.to_legacy()

pcd = pcd.voxel_down_sample(voxel_size=0.01)
inc,PCL= Segmentation(pcd)
o3d.visualization.draw_geometries([pcd])

pickle_file = "C:\\data_for_learning\\RegressionGrasp.pickle" ### Write path for the full_shape_val_data.pkl file ###
with open(pickle_file, 'rb') as f:
    reg = pickle.load(f)

Clus=Clustering(pcd,eps=0.008, min_points=20)
for i in range(len(Clus)):
    Clus[i].estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*5, max_nn=300))

    fph=o3d.pipelines.registration.compute_fpfh_feature(Clus[i], o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*11, max_nn=500))
    L=CenterOfPCD(np.asarray(Clus[i].points))
    fph = np.array(np.asarray(fph.data)).T
    fph=np.append(fph,L,axis=1)

    aff=reg.predict(fph)
    aff=np.asarray(aff).reshape((len(Clus[i].points),1))
    Clus[i].colors=o3d.utility.Vector3dVector(np.concatenate((aff,np.asarray(np.asarray(Clus[i].colors)[:, :1]),np.asarray(np.asarray(Clus[i].colors)[:, 1:2])),axis=1))
    o3d.visualization.draw_geometries([Clus[i]])

#pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*5, max_nn=50))
#fph=o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*11, max_nn=100))
#L=CenterOfPCD(np.asarray(pcd.points))
#for i in range(len(Clus)):
#    o3d.visualization.draw_geometries([Clus[i]])

#print(intr.fx, 0, intr.ppx, 0, intr.fy, intr.ppy,0, 0, 1)

if False:
    from tomlkit import inline_table




    def visualizer(rgb,Depth,hasrun=[]):
        
        if not hasrun:
            hasrun.append("second time")
            plt.ion()
            DEPTH = plt.subplot(1,2,1)
            RGB = plt.subplot(1,2,2)
            global im1
            global im2
            im2=RGB.imshow(rgb)
            im1=DEPTH.imshow(Depth)


        else:
            im2.set_data(rgb)
            im1.set_data(Depth)
            plt.show()
            plt.pause(0.001)

    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
    cam = o3d.camera.PinholeCameraIntrinsic()
    print(o3d.io.read_pinhole_camera_intrinsic("real_sense_intrinsic").intrinsic_matrix)
    print(o3d.io.read_pinhole_camera_intrinsic("real_sense_intrinsic").intrinsic_matrix)

    intrinsic=o3d.io.read_pinhole_camera_intrinsic("real_sense_intrinsic").intrinsic_matrix
    intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[231.7225,0,161.8237],[0,229.9110,80.4734],[0,0,1.0000]])))
    #[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd,intrinsic)

    o3d.visualization.draw_geometries([pcd])


    def DefinePointCloud(img,Voxel_Choise,intrinsicMatrix):
        PCL = o3d.t.geometry.PointCloud.create_from_rgbd_image(img,intrinsic)
        PCL=o3d.t.geometry.PointCloud.to_legacy(PCL)
        if Voxel_Choise==True:
            voxel_down_pcd = PCL.voxel_down_sample(voxel_size=0.02)
            voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
            return voxel_down_pcd

        else: 
            return PCL

    def Segmentation(pcl):
        plane_model, inliers = pcl.segment_plane(distance_threshold=0.20,
                                            ransac_n=3,
                                            num_iterations=2000)       
        inlier_cloud = pcl.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcl.select_by_index(inliers, invert=True)

        return inlier_cloud,outlier_cloud

    def Clustering(pcl,eps,min_points):
        labels = np.array(pcl.cluster_dbscan(eps, min_points,False))
        #dbscan labeller alle point clouds. Man finder den point cloud med højeste label. En cluster er alle de points med samme label. 
        max_label = labels.max()
        Clus=[]


        for i in range(max_label+1):
            id=np.where(labels==i)[0]
            pcl_i=pcl.select_by_index(id)
            Clus.append(pcl_i)

        return Clus

    ###Main loop

    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
    PCL=DefinePointCloud(im_rgbd,True,intrinsic)
    inc,PCL= Segmentation(PCL)
    #eps lav en cirkel med radius af 30 cm hvis der er 425 obj i så er det en cluster. 
    Clus=Clustering(PCL,eps=0.2, min_points=202)

    # Features from each object 

    #Clus[0].estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=30))
    #fph=o3d.pipelines.registration.compute_fpfh_feature(Clus[0], o3d.geometry.KDTreeSearchParamHybrid(radius=0.02*2, max_nn=50))
    o3d.visualization.draw_geometries(Clus)


    #visualizer(im_rgbd.depth,im_rgbd.color)


    #if msvcrt.kbhit():
    #    if msvcrt.getwche() == 'c':
        #   break






    plt.ioff() # due to infinite loop, this gets never called.
    plt.close()
