# -*- coding: utf-8 -*-
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
attributeNames = ['num_preg', 'glucose', 'pressure', 'thickness', 'insulin', 'bmi', 'function', 'age', 'class']
myData = pandas.read_csv(url, names=attributeNames)
print(myData.head(20))
# Change layout to fit all variables
myData.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()
myData.hist()
plt.show()
scatter_matrix(myData)
plt.show()
valueArray = myData.values

#RUN A
print("RUN A")
# Normalize Data
X = preprocessing.normalize(valueArray[:,0:8])
Y = valueArray[:,8]
test_size = 0.20
seed = 7
X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = []
for name, model in models:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# Get Accuracy Scores for all 4 Classifiers
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knnpredictions = knn.predict(X_validate)
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
cartpredictions = cart.predict(X_validate)
nb = GaussianNB()
nb.fit(X_train, Y_train)
nbpredictions = nb.predict(X_validate)
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validate)
print()
print(accuracy_score(Y_validate, knnpredictions))
print(accuracy_score(Y_validate, cartpredictions))
print(accuracy_score(Y_validate, nbpredictions))
print(accuracy_score(Y_validate, predictions))
print(confusion_matrix(Y_validate, predictions))
print(classification_report(Y_validate, predictions))

#RUN B
print("RUN B")
# Normalize Data
X = preprocessing.normalize(valueArray[:,[1,2,3,5]])
Y = valueArray[:,8]
test_size = 0.20
seed = 7
X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = []
for name, model in models:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# Get Accuracy Scores for all 4 Classifiers
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knnpredictions = knn.predict(X_validate)
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
cartpredictions = cart.predict(X_validate)
nb = GaussianNB()
nb.fit(X_train, Y_train)
nbpredictions = nb.predict(X_validate)
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validate)
print()
print(accuracy_score(Y_validate, knnpredictions))
print(accuracy_score(Y_validate, cartpredictions))
print(accuracy_score(Y_validate, nbpredictions))
print(accuracy_score(Y_validate, predictions))
print(confusion_matrix(Y_validate, predictions))
print(classification_report(Y_validate, predictions))
