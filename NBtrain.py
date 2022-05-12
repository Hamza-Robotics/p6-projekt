import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn import svm
import time
import numpy as np
import pickle
import os

x = np.load('C:\\data_for_learning\\x_values.npy')
y = np.load('C:\\data_for_learning\\y_values.npy')

#[32..features]:
X=x
#[Grasp Affordance, Wrap Affordance...]
Y=y

print(np.shape(y))


Reg = KNeighborsRegressor(n_neighbors=158)
Reg.fit(X, y)
print("done learning")

#print(Reg.predict(X))

#Pickle
pickle_out = open("C:\\data_for_learning\\RegressionNB.pickle","wb")
pickle.dump(Reg, pickle_out)
pickle_out.close()


#FINISHED SOUND
#os.system('Murdar.mp3')

print("done")