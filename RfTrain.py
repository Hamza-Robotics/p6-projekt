from sklearn.ensemble import RandomForestRegressor
import time
import numpy as np
import pickle
import os

x = np.load('x_values.npy')
y = np.load('y_values.npy')

#[32..features]:
X=x
#[Grasp Affordance, Wrap Affordance...]
Y=y
Reg=RandomForestRegressor(n_estimators=100)
print("starting to learn")
Reg=Reg.fit(X,Y)
print("done learning")

#print(Reg.predict(X))

#Pickle
pickle_out = open("Regression.pickle","wb")
pickle.dump(Reg, pickle_out)
pickle_out.close()


#FINISHED SOUND
os.system('Murdar.mp3')

print("done")