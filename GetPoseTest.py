import sys
import urx
import time
import numpy as np

rob = urx.Robot("172.31.1.115")
test = rob.get_pos()
testArray = np.asarray([test[0],test[1],test[2]])
#npTestArray = np.asarray(test[0],test[1],test[2])
print(test)
print(testArray)

print(type(test))
print(type(testArray))
print(np.asarray(test))
print(np.shape(np.asarray(test)))
print(np.shape(np.asarray(testArray)))
print(np.asarray(testArray))