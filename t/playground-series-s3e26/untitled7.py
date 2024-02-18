# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:08:40 2024

@author: kgavahi
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import numpy as np
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('train.csv')

le = LabelEncoder()


df['Drug'] = le.fit_transform(df['Drug'])
df['Ascites'] = le.fit_transform(df['Ascites'])
df['Hepatomegaly'] = le.fit_transform(df['Hepatomegaly'])
df['Spiders'] = le.fit_transform(df['Spiders'])
df['Edema'] = le.fit_transform(df['Edema'])

df['Sex'] = le.fit_transform(df['Sex'])
df['Status'] = le.fit_transform(df['Status'])


features = ['N_Days', 'Drug', 'Age', 'Sex', 'Ascites', 'Hepatomegaly',
       'Spiders', 'Edema', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
       'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin',
       'Stage']

target = 'Status'


# Split the data into training and testing sets
X = df[features]
y = df[target]

n = len(X)

X_train = X[:int(n*0.85)]
X_test = X[int(n*0.85):]
y_train = y[:int(n*0.85)]
y_test = y[int(n*0.85):]
X_train, y_train= shuffle(X_train, y_train, random_state=42)



import seaborn as sns




# Normalize the data
mean_train_X = X_train.mean()
mean_train_y = y_train.mean()

std_train_X = X_train.std()
std_train_y = y_train.std()


X_train = (X_train - mean_train_X)/ std_train_X
X_test = (X_test - mean_train_X)/ std_train_X

#y_train = (y_train - mean_train_y)/ std_train_y
#y_test = (y_test - mean_train_y)/ std_train_y


sns.violinplot(data=y_train)


#
forest_para = {'n_estimators':[100], 
               'max_depth':[4],'min_samples_leaf':[4], 
               'random_state':[42]}

        
forest_cl = RandomForestClassifier()


# Cross validation 5 fold
grid_search = GridSearchCV(forest_cl, forest_para, cv=5, scoring='neg_mean_squared_error',
return_train_score=True)


grid_search.fit(X_train, y_train)


print(grid_search.best_params_)


best_model = grid_search.best_estimator_

y_pred_pr = best_model.predict_proba(X_test)
y_pred = best_model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
#ConfusionMatrixDisplay(confusion_matrix=cm).plot();

accuracy = accuracy_score(y_test, y_pred)



print("Accuracy:", accuracy)


# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();

y_test_oh = np.array(pd.get_dummies(y_test))


    
s=0    
for i in range(len(y_pred)):
    row_sum = 0
    for j in range(3):
        
        row_sum+= np.log(y_pred_pr[i, j]) * y_test_oh[i, j]
        
    s+=row_sum

logloss = -1/len(y_pred_pr) * s
print('logloss = ', logloss)





