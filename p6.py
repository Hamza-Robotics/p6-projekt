import numpy as np

mtx=(np.asarray([[380.416,0,317.8883],[0,380.416, 241.048 ],[0 , 0, 1]]))
print(mtx)
np.save("RealSenseCameraParams\\RealSenseCameraParmas640x480.npy",mtx)

mtx=(np.asarray([[951.040,0,954.709],[0,951.040,542.619 ],[0 , 0, 1]]))
print(mtx)
np.save("RealSenseCameraParams\\RealSenseCameraParmas1980x1080.npy",mtx)


mtx=(np.asarray([[634.027,0,636.472],[0,634.027,361.746 ],[0 , 0, 1]]))
mtx=(np.asarray([[634.027,0,361.746],[0,634.027, 636.472 ],[0 , 0, 1]]))

print(mtx)
np.save("RealSenseCameraParams\\RealSenseCameraParmas1280x720.npy",mtx)

