import numpy as np
import scipy.io
import cv2
import open3d as o3d
import os 

path='C:\\data_for_learning\\tools\\'

class object(): 
    # default constructor
    def __init__(self,num,type,datapoints):
        self.number=num
        self.type=type
        self.points=datapoints

        if len(self.points) !=num:
            raise ValueError("number does not match observation")

    def path(self):
        list=[]   #list til hovedsti
        for i in range(self.number):
            string=path+self.type+'_0'+str(i)
            list.append(string)
        return list

    def extractData(self):
        list=[]
        for i in range(self.number-1):
            if i<9:
                list.append(path+self.type+"_0"+str(i+1))   #C:\data_for_learning\tools\knife_11
            else:
                list.append(path+self.type+"_"+str(i+1))
        All_rgb=[]
        All_AFF=[]
        All_DEP=[]

        for dir in list:
            rgb=[]
            depth=[]
            aff=[]
            for file in os.listdir(dir):
                filename = os.fsdecode(file)
                if filename.endswith(".jpg"):
                    rgb.append(cv2.imread(dir+"\\"+filename))
                if filename.endswith(".png"):
                    depth.append(cv2.imread(dir+"\\"+filename, cv2.IMREAD_GRAYSCALE))
                if filename.endswith("_label_rank.mat"):
                    mat = scipy.io.loadmat(dir+"\\"+filename)
                    m=((mat['gt_label']))
                    aff.append(m[:,:,1])
            
            All_AFF.append(aff)
            All_DEP.append(depth) 
            All_rgb.append(rgb)
       
        return All_rgb, All_AFF, All_DEP

#knife=object(12,"knife",[242,271,205,365,285,314,233,331,261,242,340,359])
knife=object(2,"knife",[242,271])

rgb,aff,dep=knife.extractData()

fx,fy=525,525
cx,cy=320,240



intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])))
intrinsic=o3d.camera.PinholeCameraIntrinsic(640,480 ,fx,fy,cx,cy)

for j in range(len(dep)):
    for i in range(len(dep[j])):
        cv2.imshow("Img",(dep[j][i])*25)
        print(np.max(dep[j][i]),np.min(dep[j][i]))

        cv2.waitKey(1)
        Dep=o3d.geometry.Image(np.asarray(dep[j][i]+aff[j][i]).astype(np.uint8))
        Aff=o3d.geometry.Image(np.asarray(rgb[j][i]).astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(Aff,Dep)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image((rgbd),(intrinsic))
        #o3d.visualization.draw_geometries([pcd])




if False:
    for j in range(len(aff)):
        for i in range(len(aff[j])):
            cv2.imshow("img",(aff[j][i]))

            print(np.max((aff[j][i])))
            cv2.waitKey(10)