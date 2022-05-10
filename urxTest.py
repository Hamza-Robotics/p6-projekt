import sys
import urx
import time
#from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

a=0.4
v=0.05
rob = urx.Robot("172.31.1.115")
with open('gripper-close2.script', 'r') as script:
    rob.send_program(script.read())
    time.sleep(10)
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(2, (0, 0, 0.1))
time.sleep(0.2)  #leave some time to robot to process the setup commands
rob.movej((0, 0, 0, 0, 0, 0), a, v, relative=True)
#rob.movel((x, y, z, rx, ry, rz), a, v)
print("Current tool pose is: ",  rob.get_pose())
#rob.movel((0.1, 0, 0, 0, 0, 0), a, v, relative=True)  # move relative to current pose
#rob.translate((0.1, 0, 0))#, a, v)  #move tool and keep orientation
rob.stopj(a)

rob.movel((0, 0, 0.10, 0, 0, 0), wait=True, relative=True)
#while True :
#    time.sleep(0.1)  #sleep first since the robot may not have processed the command yet
#    if rob.is_program_running():
#        break
print("Current tool pose is: ",  rob.get_pose())

rob.movel((0, 0.10, 0, 0, 0, 0), wait=True, relative=True)
#while rob.get_force() < 50 :
#    time.sleep(0.01)
#    if not rob.is_program_running():
#        break
rob.stopl()
print("Current tool pose is: ",  rob.get_pose())

try:
    rob.movel((0,0,0.1,0,0,0), relative=True)
except RobotError as ex:
    print("Robot could not execute move (emergency stop for example), do something", ex)
