# KNN is used to classify data(ie: features) into certain categories.
# K is an odd number so that the nearest "votes" are in favor of one side.
# If k is too many values then it could go to the group with the most votes but not closest.
# This project utilizes data of Iris plants from the UCI Machine Learning Repository
# and classifies the different species of Iris flowers by the length and width of the
# sepals and petals.

import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import linear_model, preprocessing

data = pd.read_csv("iris.data")
print(data.head())

le = preprocessing.LabelEncoder()
sl = le.fit_transform(list(data["sepallength"]))
sw = le.fit_transform(list(data["sepalwidth"]))
pl = le.fit_transform(list(data["petallength"]))
pw = le.fit_transform(list(data["petalwidth"]))
cls = le.fit_transform(list(data["cls"]))
# take the labels and turn them into integers
# returns buying into integer values
# print(sl)

predict = "cls"

X = list(zip(sl, sw, pl, pw))
y = list(cls)
# convert into one big list
# turn into array

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("Accuracy: ", acc)

predicted = model.predict(x_test)
names = ["Setosa", "Versicolour", "Virginia"]
# names list so that we can convert int predictions into string reps


for x in range(len(predicted)):
    print("\nPredicted: " + names[predicted[x]], "\nData: ", x_test[x], "\nActual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("\nN: ", n)
    # shows the neighbors of each point in our testing data

