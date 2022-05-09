import sklearn
from sklearn.ensemble import RandomForestRegressor
import sklearn.feature_selection
import time
import numpy as np
import pickle
import os

x = np.load('C:\\data_for_learning\\x_values.npy')
y = np.load('C:\\data_for_learning\\y_values.npy')

#[32..features]:
X=x
#[[Grasp Affordance, Wrap Affordance],[..,..],...]
Y=y

print("x=",np.shape(x),"   y;",np.shape(Y))

#Feature Selection:
#checking for low varriance (ingen features skal fjernes her)
sel = sklearn.feature_selection.VarianceThreshold(threshold=(.7 * (1 - .7)))
sel.fit_transform(X)





Reg=RandomForestRegressor(n_estimators=250, random_state=42,verbose=2,max_depth=40,n_jobs=-1)

print(np.shape(x),np.shape(y))
print(np.shape(X),np.shape(Y))

print("starting to learn")
Reg=Reg.fit(X,Y)
print("done learning")

pickle_out = open("C:\\data_for_learning\\RegressionGrasp.pickle","wb")
pickle.dump(Reg, pickle_out)
pickle_out.close()


#FINISHED SOUND
os.system('Murdar.mp3')

print("done")