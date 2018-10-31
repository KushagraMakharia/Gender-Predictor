from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
Treeclf = tree.DecisionTreeClassifier()
SVCclf = LinearSVC()
NBclf = GaussianNB()


# CHALLENGE - create 3 more classifiers...
# 1 DecisionTreeClasifier
# 2 Linear Support Vector Machine
# 3 Gaussian Naive Bayes

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
Treeclf = Treeclf.fit(X, Y)
SVCclf = SVCclf.fit(X, Y)
NBclf = NBclf.fit(X,Y)


prediction1 = Treeclf.predict(X)
acc_tree = accuracy_score(Y, prediction1) * 100
prediction2 = SVCclf.predict(X)
acc_SVC = accuracy_score(Y, prediction2) * 100
prediction3 = NBclf.predict(X)
acc_NB = accuracy_score(Y, prediction3) * 100


# CHALLENGE compare their reusults and print the best one!

print("Accuracy of DecisionTreeClasifier: ", acc_tree)
print("Accuracy of Support Vector Machine: ", acc_SVC)
print("Accuracy of Naive Bayes: ", acc_NB)


index = np.argmax([acc_tree, acc_SVC, acc_NB])
classifiers = {0: 'Decision Tree', 1: 'Support Vector Machine', 2: 'Naive Bayes'}
print('Best gender classifier is {}'.format(classifiers[index]))
