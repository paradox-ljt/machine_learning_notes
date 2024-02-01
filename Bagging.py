from sklearn.model_selection import cross_val_score
from sklearn import datasets
# prepare data
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
# Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
scores = cross_val_score(bagging, X, y)
print('Bagging-accuracy:',scores.mean())