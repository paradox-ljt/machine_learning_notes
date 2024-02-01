import numpy as np
import pandas as pd
from numpy.linalg import eig
import seaborn as sns
import matplotlib.pyplot as plt

def pca(X,k):
    X = X - X.mean(axis = 0) #向量X去中心化
    X_cov = np.cov(X.T, ddof = 0) #计算向量X的协方差矩阵，自由度可以选择0或1
    eigenvalues,eigenvectors = eig(X_cov) #计算协方差矩阵的特征值和特征向量
    klarge_index = eigenvalues.argsort()[-k:][::-1] #选取最大的K个特征值及其特征向量
    k_eigenvectors = eigenvectors[klarge_index] #用X与特征向量相乘
    return np.dot(X, k_eigenvectors.T)

excel = pd.read_excel(r"C:\Users\华为\Desktop\程序\主成分分析.xlsx")
X = np.array(excel)
k = 2
X_pca = pca(X, k)
print(X_pca)

X = X - X.mean(axis = 0)

#计算协方差矩阵
X_cov = np.cov(X.T, ddof = 0)

#计算协方差矩阵的特征值和特征向量
eigenvalues,eigenvectors = eig(X_cov)

tot = sum(eigenvalues)
var_exp = [(i/tot) for i in sorted(eigenvalues, reverse = True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,7), var_exp, alpha = 0.5, align = 'center', label = 'individual var')
plt.step(range(1,7), cum_var_exp, where = 'mid', label = 'cumulative var')
plt.ylabel('variance rtion')
plt.xlabel('principal components')
plt.legend(loc = 'best')
plt.show()