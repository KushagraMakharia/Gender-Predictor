from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
clf = tree.DecisionTreeClassifier()
SVCclf = LinearSVC()
NBclf = GaussianNB()


# CHALLENGE - create 3 more classifiers...
# 1 DecisionTreeClasifier
# 2 Linear Support Vector Macine
# 3 Gaussian Naive Bayes

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
SVCclf = SVCclf.fit(X, Y)
NBclf = NBclf.fit(X,Y)

prediction1 = clf.predict([[160, 60, 38]])
prediction2 = SVCclf.predict([[160, 60, 38]])
prediction3 = NBclf.predict([[160, 60, 38]])


# CHALLENGE compare their reusults and print the best one!

print(prediction1)
print(prediction2)
print(prediction3)
