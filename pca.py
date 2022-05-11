import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)
x = np.load('C:\\data_for_learning\\x_values.npy')
y = np.load('C:\\data_for_learning\\y_values.npy')
x, X_test, y, y_test = train_test_split(x, y, random_state=42)

y=(y > 0).astype(int)

x = StandardScaler().fit_transform(x)

pca = decomposition.PCA(n_components=3)
pca.fit(x)
X = pca.transform(x)

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

ax.scatter(x[:, 0], x[:, 1],x[:, 2],
            c=y, edgecolor='none', alpha=0.5)

plt.show()