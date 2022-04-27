
import numpy as np
from sklearn import preprocessing
import pickle

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

x = np.load('C:\\data_for_learning\\x_values.npy')
y = np.load('C:\\data_for_learning\\y_values.npy')

#[32..features]:
#X=scale(x)
#[Grasp Affordance, Wrap Affordance...]
#Y=scale(y)

X=x
Y=y



print(np.shape(X))
print(np.shape(Y))


Reg=make_pipeline(StandardScaler(),SGDRegressor(max_iter=10000, tol=1e-3))





print("starting to learn")
Reg.fit(X,Y)
print("done learning")



pickle_out = open("C:\\data_for_learning\\RegressionGD.pickle","wb")
pickle.dump(Reg, pickle_out)
pickle_out.close()


#FINISHED SOUND
#os.system('Murdar.mp3')

print("done")