from sklearn.model_selection import cross_val_score
from sklearn import datasets
# prepare data
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
# AdaBoosting
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, X, y)
print('AdaBoost-accuracy:',scores.mean())