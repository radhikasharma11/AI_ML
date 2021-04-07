from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy

iris = load_iris()
print(type(iris))
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)
print(iris.data.shape)

X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=4)

scores = {}
scores_list = []
k_range = range(1,26)

# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train,y_train)
#     y_pred = knn.predict(X_test)
#     # print(y_pred)
#     # print('-'*150)
#     # print(y_test)
#
#     scores[k] = metrics.accuracy_score(y_test,y_pred)
#     scores_list.append(metrics.accuracy_score(y_test,y_pred))
# plt.plot(k_range,scores_list)
# plt.xlabel("Value of K for kNN")
# plt.ylabel('Testing Accuracy')
# plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
y_pred = knn.predict([[5,4,2,2]])
print(y_pred)
# print('-'*150)
# print(y_test)