import numpy as np
import open3d as o3d
import json 
import pyrealsense2 as rs
print("1200")
print(np.load("Calibration__Data\\intrinicmat1980x1200.npy"))


print("   640 ")


print(np.load("Calibration__Data\\intrinicmat640xo480.npy"))


print("\n\n\n")

with open("CameraSetup.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))




##Startting camera
pipeline = rs.pipeline()
cfg = pipeline.start() # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0)
intrinsic=o3d.camera.PinholeCameraIntrinsic(intr.width,intr.height ,intr.fx,intr.fy,intr.ppx,intr.ppy)

print(intrinsic.intrinsic_matrix)