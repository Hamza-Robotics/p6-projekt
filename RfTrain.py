from sklearn.ensemble import RandomForestRegressor
import time
import pickle
import os



#[32..features]:
X=[[0, 0], [1, 1]]
#[Grasp Affordance, Wrap Affordance...]
Y=[1,2]
Reg=RandomForestRegressor(n_estimators=100)

Reg=Reg.fit(X,Y)

print(Reg.predict(X))

#Pickle
pickle_out = open("Regression.pickle","wb")
pickle.dump(Reg, pickle_out)
pickle_out.close()


#FINISHED SOUND
os.system('Murdar.mp3')