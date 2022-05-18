from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os

x = np.load('C:\\data_for_learning\\x_values_c.npy')
y = np.load('C:\\data_for_learning\\y_values_c.npy')

#[32..features]:
X=x
#[[Grasp Affordance, Wrap Affordance],[..,..],...]
Y=y
y[y>0]=1


print("x=",np.shape(x),"   y;",np.shape(Y))


clf = RandomForestClassifier(n_estimators=100,max_depth=30,random_state=43,verbose=2,n_jobs=-1)

print(np.shape(x),np.shape(y))
print(np.shape(X),np.shape(Y))

print("starting to learn")
Clf=clf.fit(X,Y)
print("done learning")

pickle_out = open("C:\\data_for_learning\\RFCLASS.pickle","wb")
pickle.dump(Clf, pickle_out)
pickle_out.close()


#FINISHED SOUND
os.system('Murdar.mp3')

print("done")