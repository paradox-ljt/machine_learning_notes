from sklearn.model_selection import cross_val_score
from sklearn import datasets
# prepare data
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
# Gradient Tree Boosting
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
scores = cross_val_score(clf, X, y)
print('GDBT-accuracy:',scores.mean())