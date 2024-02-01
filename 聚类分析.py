import pandas as pd
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
excel = pd.read_excel(r"C:\Users\华为\Desktop\程序\聚类分析.xls")
x = np.array(excel)

rel = pd.DataFrame()
table = None
color_map = ['k', 'pink', 'r', 'y', 'g']

def draw(titile, table):
    fig = plt.figure()  # 建立一个空间
    ax = fig.add_subplot(111, projection='3d')  # 3D坐标
    for i in range(114):
        ax.scatter(x[i][0],x[i][1],x[i][2], color=color_map[table[i]])
    plt.show()

model = sc.KMeans(n_clusters=5)# k-means 聚类
yhat = model.fit_predict(x)
table = model.predict(x)
print(table)
rel['KMeans'] = table
draw("KM", table)

model = AgglomerativeClustering(n_clusters=5) #层次聚类
yhat = model.fit_predict(x)
# table = model.predict(x)
# print(table)
# rel['AgglomerativeClustering'] = table
draw("AC", table)

model = Birch(threshold=0.01, n_clusters=5) #Birch
yhat = model.fit_predict(x)
table = model.predict(x)
print(table)
rel['Birch'] = table
draw("Birch", table)

model = MiniBatchKMeans(n_clusters=5) # Mini-Batch K-均值（K-均值修改版)
yhat = model.fit_predict(x)
table = model.predict(x)
print(table)
rel['MiniBatchKMean'] = table
draw("MiniBatchKMean", table)

model = SpectralClustering(n_clusters=5) # 光谱聚类
# yhat = model.fit_predict(x)
# table = model.predict(x)
# print(table)
# rel['SpectralClustering'] = table
draw("SpectralClustering", table)

model = GaussianMixture(n_components=5) # 高斯混合模型
yhat = model.fit_predict(x)
table = model.predict(x)
print(table)
rel['GaussianMixture'] = table
draw("GaussianMixture", table)