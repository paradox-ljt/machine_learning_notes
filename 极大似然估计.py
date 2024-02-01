from sklearn import datasets
iris = datasets.load_iris() 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(iris.data, iris.target)
y_pred=clf.predict(iris.data)
accuracy = (iris.target != y_pred).sum() / iris.data.shape[0]
print("MLE-Number of samples %d wrong-sample : %d accuracy: %f" % (iris.data.shape[0],(iris.target != y_pred).sum(), accuracy))