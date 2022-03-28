import matplotlib.pyplot as plt
import numpy as np

imgd=np.load('Bagfile.txt.npy')

def visualizer(depth_Image,hasrun=[]):

    if not hasrun:
        hasrun.append("second time")
        plt.ion()
        DEPTH = plt.subplot()
        global im1
        im1=DEPTH.imshow(depth_Image)
        plt.pause(0.001)

    else:
        im1.set_data(depth_Image)
        plt.pause(0.001)


for i in range(len(imgd)):
    visualizer(imgd[i])
