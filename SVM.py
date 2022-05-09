from sklearn import pipeline
from sklearn.pipeline import Pipeline 
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  f_regression
import numpy as np
from sklearn.utils import shuffle
import pickle
import os
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import LinearSVR



x = np.load('C:\\data_for_learning\\x_values.npy')
y = np.load('C:\\data_for_learning\\y_values.npy')
X, Y= shuffle(x, y,random_state=0)




# Create and fit selector
selector = SelectKBest(f_regression, k=10)
selector.fit(X, Y)
# Get columns to keep and create new dataframe with those only
xn = selector.get_support(1)
print(xn)

Xn=(X[:,xn])

svr = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])


C=[1,50,200]
epsilon=[0.1, 0.5]
kernel=['linear']


# Create the random grid
random_grid = {'svr__C': C,
            'svr__epsilon': epsilon,
            'svr__kernel': kernel           
}


Reg = RandomizedSearchCV(estimator = svr, param_distributions = random_grid, n_iter = 12,n_jobs=-1, random_state=42)

print("starting to learn")
Reg.fit(Xn, Y)
print("done learning")




pickle_out = open("C:\\data_for_learning\\RegressionSVM.pickle","wb")
pickle.dump(Reg, pickle_out)
pickle_out.close()

print("done")

if False:

    C=[1,50,100,200]
    epsilon=[0.1,0.2,0.5]


    # Create the random grid
    random_grid = {'svr__C': C,
                'svr__epsilon': epsilon

                
    }


    svr = Pipeline([('scaler', StandardScaler()), ('svr', LinearSVR(max_iter=10000))])

    Reg = RandomizedSearchCV(estimator = svr, param_distributions = random_grid, n_iter = 9,n_jobs=-1, random_state=42)

    print("starting to learn")
    Reg.fit(X, Y)
    print("done learning")



    pickle_out = open("C:\\data_for_learning\\RegressionNEW4.pickle","wb")
    pickle.dump(Reg, pickle_out)
    pickle_out.close()


    #FINISHED SOUND
    os.system('Murdar.mp3')

    print("done")