import sys
import urx
import time
import math

rob = urx.Robot("172.31.1.115")
a=0.4
v=0.3
print("Current tcp pos is: ", rob.get_pos())


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
for i in range(5):
    rob.movej((3,0,0,0,0,0),a,v,wait = True, relative = True)
    rob.movej((-3,0,0,0,0,0),a,v,wait = True, relative = True)


rob.stop()