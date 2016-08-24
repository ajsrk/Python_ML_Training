import pandas as pd
import numpy as np
from sklearn import linear_model

import os
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

filename = os.path.join(os.path.dirname(__file__),"Datasets","data.xlsx")
#print("The file path "+filename)
data = pd.read_excel(filename)
train_set = data.ix[:15,:15]
x_train = train_set[['X']]
y_train = train_set[['Y']]
z_train = train_set[['Z']]
test_set = data[['X']].ix[15:]

# Linear Regression
model1 = linear_model.LinearRegression()
model1.fit(x_train,y_train)
model1.score(x_train,y_train)
#print('Coefficient \n'+ linear.coef_)
#print('Intercept \n' + linear.intercept_)
model1_predicted = model1.predict(test_set)
print("Linear Regression Model")
print(model1_predicted)

#Logistic Regression
model2 = linear_model.LogisticRegression()
model2.fit(x_train,z_train)
model2.score(x_train,z_train)
model2_predicted = model2.predict(test_set)
print("Logistic Regression Model")
print(model2_predicted)

#Decision Tree
from sklearn import tree
model3 = tree.DecisionTreeClassifier()
model3.fit(x_train,z_train)
model3.score(x_train,z_train)
model3_predicted = model3.predict(test_set)
print("Decision Tree Classification Model")
print(model3_predicted)

model4 = tree.DecisionTreeRegressor()
model4.fit(x_train,y_train)
model4.score(x_train,y_train)
model4_predicted = model4.predict(test_set)
print("Decision Tree Regression Model")
print(model4_predicted)

#Support Vector Machine
from sklearn import svm
model5 = svm.SVC()
model5.fit(x_train,z_train)
model5.score(x_train,z_train)
model5_predicted = model5.predict(test_set)
print("SVM Classifier Model")
print(model5_predicted)

#Naive Bayes
from sklearn import naive_bayes
model6 = naive_bayes.GaussianNB()
model6.fit(x_train,z_train)
model6_predicted = model6.predict(test_set)
print("Naive Bayes Model")
print(model6_predicted)

#K-Nearest Neighbours
from sklearn import neighbors
model7 = neighbors.KNeighborsClassifier(n_neighbors=6)#default value of n_neighbours is 6
model7.fit(x_train,z_train)
model7_predicted = model7.predict(test_set)
print("K-Nearest Neighbours Model")
print(model7_predicted)

#K-Means
from sklearn import cluster
model8 = cluster.KMeans(n_clusters=2, random_state=0)
model8.fit(x_train)
model8_predicted =  model8.predict(test_set)
print("K-Means Model")
print(model8_predicted)

#Random Forest
from sklearn import ensemble
model9 = ensemble.RandomForestClassifier()
model9.fit(x_train,z_train)
model9_predicted = model9.predict(test_set)
print("Random Forest Classifier Model")
print(model9_predicted)

#Dimensionality Reduction
from sklearn import decomposition
pca = decomposition.PCA()
factor = decomposition.FactorAnalysis()
x_train_reduced = pca.fit_transform(x_train)
test_set_reduced = pca.fit_transform(test_set)
print("Dimensionality Reduction")
print('Train reduced ')
print(x_train_reduced)
print('Test reduced ')
print(test_set_reduced)
model10 = ensemble.RandomForestClassifier()
model10.fit(x_train_reduced,z_train)
model10_predicted = model10.predict(test_set_reduced)
print(model10_predicted)

#Gradient Boost and AdaBoost
from sklearn import ensemble
model11 = ensemble.GradientBoostingClassifier(n_estimators=100,learning_rate=1,max_depth=1, random_state=0)
model11.fit(x_train,z_train)
model11_predicted = model11.predict(test_set)
print("Gradient Boost")
print(model11_predicted)
