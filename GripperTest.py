import sys
import urx
import time
import numpy as np
import pickle
import math
import math3d as m3d

rob = urx.Robot("172.31.1.115")
a=0.4
v=0.5
startingPose = rob.get_pose()
print("Starting Pose tcp pose is: ", startingPose)


#print("current orientation is: ", startingPose.get_orient())


#rob.movej((0, -math.pi/3, math.pi/3, 0, math.pi/2, math.pi/2), a, v, wait=True, relative=False)
#rob.movej([0.6676359176635742, -0.7742732206927698, 0.7994108200073242, 1.4085272550582886, 1.6080025434494019, 1.4363412857055664], a, v, wait=True, relative=False)
def horizintalMovements():
    rob.movej([0.11325842142105103, -0.38410789171327764, 0.6964402198791504, 1.9883853197097778, 0.7105001211166382, 0.6198690533638], a, v, wait=True, relative=False)
    time.sleep(4)
    print("Current joint poses are: ", rob.getj())
    #print("Current gripper pose is: ", rob.get_orientation())

    for i in range(8):
        rob.movel((0.01, -0.01, 0, math.pi/20, math.pi/20, 0), a, v, wait=True, relative=True)
        print("Current joint poses are: ", rob.getj())
        time.sleep(4)

def VerticalMovements():
    rob.movej([0.7853862047195435, -1.515883747731344, 2.183744430541992, 0.028925776481628418, 1.7645397186279297, 1.702071189880371], a, v, wait=True, relative=False)
    time.sleep(4)
    print("Current joint poses are: ", rob.getj())

    for i in range(5):
        rob.movel((0, 0, 0.15, math.pi/(40+i*10), -math.pi/(40+i*10), 0), a, v, wait=True, relative=True)
        print("Current joint poses are: ", rob.getj())
        time.sleep(4)

def moveIngremental(begin = False, vertical=False):
    if vertical and begin:
        rob.movej([0.7853862047195435, -1.515883747731344, 2.183744430541992, 0.028925776481628418, 1.7645397186279297, 1.702071189880371], a, v, wait=True, relative=False)
    elif vertical:
        rob.movel((0, 0, 0.15, math.pi/(40+i*10), -math.pi/(40+i*10), 0), a, v, wait=True, relative=True)
    elif begin:
        rob.movej([0.11325842142105103, -0.38410789171327764, 0.6964402198791504, 1.9883853197097778, 0.7105001211166382, 0.6198690533638], a, v, wait=True, relative=False)
    else:
        rob.movel((0.01, -0.01, 0, math.pi/20, math.pi/20, 0), a, v, wait=True, relative=True)
    time.sleep(4)

def dontdothis():
    for i in range(5):
        rob.movej((3,0,0,0,0,0),a,v,wait = True, relative = True)
        rob.movej((-3,0,0,0,0,0),a,v,wait = True, relative = True)

def tryThis():
    mytcp = m3d.Transform()  # create a matrix for our tool tcp
    #mytcp.pos.z = 0.18
    mytcp.pos = (-0.4,-0.4,0.2)
    mytcp.orient.rotate_xb(math.pi)
    #mytcp.orient.rotate_yb(math.pi)
    mytcp.orient.rotate_zb(math.pi/2)
    print(mytcp)
    approach = mytcp.get_orient() * m3d.Vector(0,0,-0.1)
    print(approach)
    mytcp.pos += approach
    print(mytcp)
    rob.set_pose(mytcp,a,v,wait = True, command = 'movej')
    mytcp.get_matrix()

    print("Current tcp pose is: ", rob.get_pose())
    with open('gripperScripts//openg.script', 'r') as script:
        rob.send_program(script.read())
        time.sleep(2)
    rob.back(-0.1,a,v)
    print("Current tcp pose is: ", rob.get_pose())
    with open('gripperScripts//closeg.script', 'r') as script:
        rob.send_program(script.read())
        time.sleep(1)

def MoveBack():
    mytcp = m3d.Transform()  # create a matrix for our tool tcp
    mytcp.pos = (-0.46610, -0.52469, 0.09467)
    mytcp.orient = [[-0.39640639, -0.70488675, -0.58821479],[-0.4065947 ,  0.70923652, -0.57590304],[ 0.82312983,  0.01087337, -0.56774911]]

    rob.set_pose(mytcp,a,v,wait = True, command = 'movej')
    for i in range(6):
        rob.back(0.05)
        print(i)
        time.sleep(4)

def moveIngremental(begin, mode, i):
    print("moving robot")
    if mode == 0 and begin:
        rob.movej([0.7853862047195435, -1.515883747731344, 2.183744430541992, 0.028925776481628418, 1.7645397186279297, 1.702071189880371], a, v, wait=True, relative=False)
    elif mode == 0:
        rob.movel((0, 0, 0.05, np.pi/150, -np.pi/150, 0), a, v, wait=True, relative=True)
    elif mode == 1 and begin:
        rob.movej([0.11325842142105103, -0.38410789171327764, 0.6964402198791504, 1.9883853197097778, 0.7105001211166382, 0.6198690533638], a, v, wait=True, relative=False)
    elif mode == 1:
        rob.movel((0.005, -0.005, 0, np.pi/40, np.pi/40, 0), a, v, wait=True, relative=True)
    elif mode == 2 and begin:
        mytcp = m3d.Transform()
        mytcp.pos = (-0.46610, -0.52469, 0.09467)
        mytcp.orient = [[-0.39640639, -0.70488675, -0.58821479],[-0.4065947 ,  0.70923652, -0.57590304],[ 0.82312983,  0.01087337, -0.56774911]]
        rob.set_pose(mytcp,a,v,wait = True, command = 'movej')
    elif mode == 2:
        rob.back(0.05, a, v)
    elif mode == 3:
        rob.movel((-0.15, 0.15, 0, 0, 0, np.pi/20), a, v, wait=True, relative=True)
    elif mode == 4 and begin:
        mytcp = m3d.Transform()
        mytcp.pos = rob.get_pos()
        mytcp.orient = [[ 0.06740949,  0.99326777,  0.09420771],[ 0.38316714,  0.06140949, -0.9216354 ],[-0.92121599,  0.09822428, -0.37644799]]
        rob.set_pose(mytcp,a,v,wait = True, command = 'movej')
        rob.back(-0.15, a, v)
    else:
        rob.movel((0.15, -0.15, 0, 0, 0, -np.pi/18), a, v, wait=True, relative=True)

    time.sleep(1)

def whileTest():
    i=35
    while (i<54):
        print(i)
        if (i < 18):
            mode = 0
        elif (i < 35):
            mode = 1
        elif (i < 40):
            mode = 2
        elif  (i < 45):
            mode = 3
        else:
            mode = 4
        if (i==0 or i==18 or i==35 or i == 45):
            moveIngremental(True, mode, i)
        else:
            moveIngremental(False, mode, i-18)

        i=i+1
        #time.sleep(0.4)
def goFreedrive():
    while(True):
        rob.set_freedrive(1)
        time.sleep(40)

def testTCP():
    mytcp = m3d.Transform()
    mytcp.pos = (-0.4, -0.4, 0.20)
    mytcp.orient.rotate_xb(np.pi)
    rob.set_pose(mytcp,a,v,wait = True, command = 'movej')
    time.sleep(1)
    for i in range(8):
        mytcp.orient.rotate_zb(np.pi/20)
        rob.set_pose(mytcp,a,v,wait = True, command = 'movej')
        print(rob.get_pos())
        time.sleep(1)
    rob.back(-0.20)
    print(rob.get_pos())
    time.sleep(1)
    rob.movel((-0.4, -0.4, 0, np.pi, 0, 0), a, v)
    print(rob.get_pos())

def testRot():
    mytcp = m3d.Transform()
    mytcp.pos = (-0.5, -0.5, -0.055)
    mytcp.orient.rotate_xb(np.pi)
    
    rob.set_pose(mytcp,a,v,wait = True, command = 'movej')
    print("current orientation is: ", rob.get_orientation())


def testInput():
    pose_list = []
    for i in range(20):
        rob.set_freedrive(1, 60)
        input()
        pose_list.append(rob.getj())
        print(i)
    pickle_out = open("C:\\data_for_learning\\poseList.pickle","wb")
    pickle.dump(pose_list, pickle_out)
    pickle_out.close()


def testMovements():
    pickle_file = "poseList.pickle" ### Write path for the full_shape_val_data.pkl file ###
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        print(data)
    i=0
    while(i < len(data)):
        print(i)
        rob.movej(data[i],a,v)
        i+=1

def increasePickle():
    pickle_file = "poseList.pickle" ### Write path for the full_shape_val_data.pkl file ###
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    for i in range(10):
        rob.set_freedrive(1, 60)
        input()
        data.append(rob.getj())
        print("Data is now:", len(data), "entries long")
    pickle_out = open("poseList.pickle","wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

#tryThis()
#MoveBack()
#whileTest()
#goFreedrive()
#testRot()
#testInput()
#testMovements()
increasePickle()
rob.stop()