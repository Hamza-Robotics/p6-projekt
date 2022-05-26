import numpy as np
import sklearn.metrics

y_pred=np.load("ROC_DATA\\y_pred.npy")
y_actual=np.load("ROC_DATA\\y_actual.npy")
print(np.shape(y_actual),np.shape(y_pred))

import matplotlib.pyplot as plt
from sklearn.utils.multiclass import type_of_target
import scipy
#plt.plot(np.sort(y_actual)[0])
#plt.show()

y_a=np.squeeze(y_actual,axis=2)
y_p=np.squeeze(y_pred,axis=2)
print("spearman  coefficient of Correlation",scipy.stats.spearmanr(y_a.flatten(),y_p.flatten())[0])
print("R² score, the coefficient of determination", sklearn.metrics.r2_score(y_a.flatten(),y_p.flatten()))
print("Mean absolute error",sklearn.metrics.mean_absolute_error(y_a.flatten(),y_p.flatten()))
print("Mean Squared Error",sklearn.metrics.mean_squared_error(y_a.flatten(),y_p.flatten()))

#print("F1 meassure",sklearn.metrics.f1_score(y_a.flatten(),y_p.flatten()))






y_actual[0][y_actual[0]>0]=1


print(type_of_target(np.squeeze(y_actual,axis=2)[0]))

print(np.shape(np.squeeze(y_actual,axis=2)))

y_actual[y_actual>0]=1

y_ac=(y_actual.flatten())
y_pr=y_pred.flatten()


fpr, tpr, thresholds=sklearn.metrics.roc_curve(y_ac,y_pr)
print(np.shape(tpr))

print("P",sklearn.metrics.average_precision_score(y_ac,y_pr))


print(thresholds[( np.argmax(tpr - fpr))],"<--Threshold---->")

roc_auc = sklearn.metrics.auc(fpr, tpr)

t=thresholds[( np.argmax(tpr - fpr))]
y_a.flatten()[y_a.flatten()>t]=0
y_a.flatten()[y_a.flatten()<=t]=1




display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                estimator_name='Random Forrest')
display.plot()
plt.plot([0,1],[0,1],'k--')
plt.show()