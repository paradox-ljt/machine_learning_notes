import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

#Data preparation
X, y = make_circles(n_samples=1000,factor=0.5,noise=0.1)

#Gridding diagram
x0s = np.linspace(-1.5, 1.5, 100)  
x1s = np.linspace(-1.5, 1.5, 100)  
x0, x1 = np.meshgrid(x0s, x1s)  
Xtest = np.c_[x0.ravel(), x1.ravel()]  

# SVC-poly
modelSVM1 = SVC(kernel='poly', degree=3, coef0=0.2)  # 'poly' 多项式核函数
modelSVM1.fit(X, y)  
yPred1 = modelSVM1.predict(Xtest).reshape(x0.shape)  

# SVC-rbf
modelSVM2 = SVC(kernel='rbf', gamma='scale')  #'rbf' 高斯核函数
modelSVM2.fit(X, y) 
yPred2 = modelSVM2.predict(Xtest).reshape(x0.shape)  

#SVC-sigmoid
modelSVM3 = SVC(kernel='sigmoid', gamma='scale')  #'sigmoid' S型核函数
modelSVM3.fit(X, y) 
yPred3 = modelSVM3.predict(Xtest).reshape(x0.shape) 

#Classification result rendering
fig, ax = plt.subplots(figsize=(8, 6))  
ax.contourf(x0, x1, yPred1, cmap=plt.cm.brg, alpha=0.1) 
ax.contourf(x0, x1, yPred2, cmap='PuBuGn_r', alpha=0.1) 
ax.contourf(x0, x1, yPred3, cmap='CMRmap', alpha=0.1)

ax.plot(X[:,0][y==0], X[:,1][y==0], "bo")  
ax.plot(X[:,0][y==1], X[:,1][y==1], "r^") 
ax.grid(True, which='both')
ax.set_title("Classification of data")
plt.show()