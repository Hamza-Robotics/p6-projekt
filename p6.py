import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = pipeline.start() # Start pipeline and get the configuration it found
    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

    #intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[231.7225,0,161.8237],[0,229.9110,80.4734],[0,0,1.0000]])))
    intrinsic=(o3d.cpu.pybind.core.Tensor(np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])))
    rs = o3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0)
    rs.start_capture(True)  # true: start recording with capture  

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
