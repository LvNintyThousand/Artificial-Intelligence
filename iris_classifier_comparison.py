from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import matplotlib.pyplot as plt 

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

param_range_SVC = np.logspace(-5, -0.1, 20)
param_range_KNN = np.arange(1, 21, 1)

train_loss_SVC, test_loss_SVC = validation_curve(SVC(), X, Y, param_name = "gamma", param_range = param_range_SVC, scoring = "f1_weighted")
train_loss_SVC_mean = -np.mean(train_loss_SVC, axis = 1)
test_loss_SVC_mean = -np.mean(test_loss_SVC, axis = 1)

train_loss_KNN, test_loss_KNN = validation_curve(KNeighborsClassifier(), X, Y, param_name = "n_neighbors", param_range = param_range_KNN, scoring = "f1_weighted")
train_loss_KNN_mean = -np.mean(train_loss_KNN, axis = 1)
test_loss_KNN_mean = -np.mean(test_loss_KNN, axis = 1)

plt.figure()
plt.plot(param_range_SVC, train_loss_SVC_mean, marker = "o", linestyle = "-", color = "g", label = "Training")
plt.plot(param_range_SVC, test_loss_SVC_mean, marker = "o", linestyle = "-", color = "r", label = "Testing")
plt.xlabel("Gamma Values")
plt.ylabel("Loss")
plt.legend(loc = "best")
plt.show()

# best gamma value is 0.17

plt.figure()
plt.plot(param_range_KNN, train_loss_KNN_mean, marker = "o", linestyle = "-", color = "g", label = "Training")
plt.plot(param_range_KNN, test_loss_KNN_mean, marker = "o", linestyle = "-", color = "r", label = "Testing")
plt.xlabel("n_neighbors values")
plt.ylabel("Loss")
plt.legend(loc = "best")
plt.show()

# best n_neighbors value is 12.5

clf = SVC(gamma = 0.17)
clf_default = SVC()

knn = KNeighborsClassifier(n_neighbors = 12)
knn2 = KNeighborsClassifier(n_neighbors = 13)
knn_default = KNeighborsClassifier(n_neighbors = 5)

clf_R_Squared_array = []
clf_default_R_Squared_array = []
knn_R_Squared_array = []
knn2_R_Squared_array = []
knn_default_R_Squared_array = []

for i in range(1, 1001, 1):

	clf.fit(X_train, Y_train)
	clf_default.fit(X_train, Y_train)
	knn.fit(X_train, Y_train)
	knn2.fit(X_train, Y_train)
	knn_default.fit(X_train, Y_train)

	clf_R_Squared_array.append(clf.score(X_test, Y_test))
	clf_default_R_Squared_array.append(clf_default.score(X_test, Y_test))
	knn_R_Squared_array.append(knn.score(X_test, Y_test))
	knn2_R_Squared_array.append(knn2.score(X_test, Y_test))
	knn_default_R_Squared_array.append(knn_default.score(X_test, Y_test))

print("SVC R-Squared score for best gamma value : ", np.mean(clf_R_Squared_array))
print("SVC R-Squared score for default value : ", np.mean(clf_default_R_Squared_array))
print("KNN R-Squared score for 1st best n_neighbors value : ", np.mean(knn_R_Squared_array))
print("KNN R-Squared score for 2nd best n_neighbors value : ", np.mean(knn2_R_Squared_array))
print("KNN R-Squared score for default n_neighbors value : ", np.mean(knn_default_R_Squared_array))

# Conclusion: SVC() function and KNeighborsClassifier() function don't have obvious differences.
#             In some cases, SVC() performs better than KNeighborsClassifier()
#             In some cases, KNeighborsClassifier() performs better than SVC()
#             But in total SVC() performs more stable than KNeighborsClassifier()

## attempt to plot svc() function and see whether the hyper-plate comes across the outlier-cluster.

## in Iris dataset, categorical data includes a group of outliers (setosa) 