import numpy as np
import scipy.io
import cv2

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
                if filename.endswith(".png"):
                    rgb.append(cv2.imread(dir+"\\"+filename, cv2.IMREAD_GRAYSCALE))
                if filename.endswith(".jpg"):
                    depth.append(cv2.imread(dir+"\\"+filename, cv2.IMREAD_GRAYSCALE))
                if filename.endswith("_label_rank.mat"):
                    mat = scipy.io.loadmat(dir+"\\"+filename)
                    m=((mat['gt_label']))
                    aff.append(m[:,:,1])
            
            All_AFF.append(aff)
            All_DEP.append(depth) 
            All_rgb.append(rgb)
       
        return All_rgb, All_AFF, All_DEP

knife=object(12,"knife",[242,271,205,365,285,314,233,331,261,242,340,359])

rgb,aff,dep=knife.extractData()

for j in range(len(aff)):
    for i in range(len(aff[j])):
        cv2.imshow("img",(aff[j][i]))
        cv2.waitKey(10)
