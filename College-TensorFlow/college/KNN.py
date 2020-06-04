# working with irregular data
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import linear_model, preprocessing

import pickle

# KNN is a classification alg
# create groups
# data is then going to be put into closest groups ie neighbor
# set var to a nummber which will dictated the x closest points to your data point
# x closest points to var are of group A so var probably belongs to group A
# E.G x = 5, 2 are Group A 3 are Group B, var probably belongs to group B
# x has to be an odd number so there will always be an uneven split making 1 group dominant
# x should not be too high so that far off points dont get picked for small data groups
# KNN is comp heavy because distance is calc for each point every time


# note pandas reads in the first line of your data file as the col

data = pd.read_csv('car.data')
# print(data.head())

# take lables and encode to int so 
# operation can be performed 
pe = preprocessing.LabelEncoder()

# mapping data from col name to variables
buying = pe.fit_transform(list(data["buying"]))
maint = pe.fit_transform(list(data["maint"]))
door = pe.fit_transform(list(data["door"]))
persons = pe.fit_transform(list(data["persons"]))
lug_boot = pe.fit_transform(list(data["lug_boot"]))
safety = pe.fit_transform(list(data["safety"]))
clas = pe.fit_transform(list(data["class"]))
# print(buying)
predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(clas)


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

# print(x_train, y_test)
# 9 works best from what i found for this data set
model = KNeighborsClassifier(n_neighbors = 9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
# print(acc)

pre = model.predict(x_test)
names = ["unacc","acc","good","veryGood"]

for ele in range(len(pre)):
    print("Predicted: ",names[pre[ele]],"Data: ", x_test[ele], "Actual: ",names[y_test[ele]])
    # find distance of neighbors
    # model.kneighbors([x_test[ele]], 9, True)
    
    
    
'''best_model = 0
for _ in range(100):
    # runs x times
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split( x, y, test_size = 0.1)
    
    l = linear_model.LinearRegression()
    l.fit(x_train, y_train)
    accuracy = l.score(x_test, y_test)
    
    if accuracy > best_model:
        # if the model score is better than it will be saved
        best_model = accuracy
        with open("KNN.pickle", "wb") as f:
            pickle.dump(l,f)'''
    
    
    