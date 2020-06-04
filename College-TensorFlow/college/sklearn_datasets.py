import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# classes_name = ['malignant', 'benign']

# support vector machine
# a hyperplane is a line that spils/divides data
# create hyperplane splits data into classes
# rotate line to create hyperplane while having 1 point from both classes an equal distance away from the splitting line
# hyperplane is based on max distance of two points that can be reached
# distance between points is the margin
# with this margin you can classify data points more accurately 

# data data data data data data
# data data data data data data
# -------margin----margin-------
#
# ------------------hyperplane---------------------------
#
# ------margin----margin--------
# data data data data data
# data data data data data data

# for data where this wouldnt work a 2d data array can be turned into a 3d data array 
# by passing it through a kernel
# this should make it easier to implement a hyperplane but in a 3d space
# f(x1, x2) -> x3
# a soft margin allows for 

clf = svm.SVC(kernel="poly") # if you have some time to kill, will try later
# clf = svm.SVC(kernel="poly", degree=2)
# clf = svm.SVC(kernel="linear", C=2) 91%
# clf = KNeighborsClassifier(n_neighbors=10) # very high preditions even getting a 1, 9-11 neighbors seems best
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)

