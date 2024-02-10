# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:10:29 2024

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


df = pd.read_csv('train.csv')

le = LabelEncoder()


label = le.fit_transform(df['family'])

df.drop("family", axis=1, inplace=True)
 
# Appending the array to our dataFrame 
# with column name 'Purchased'
df["family"] = label

df['date'] = pd.to_datetime(df['date'])

df['DOY'] = df['date'].dt.dayofyear

df['sin'] = np.sin(df['DOY'])
df['cos'] = np.cos(df['DOY'])

#df = df[df['store_nbr']==1]








#features = ['DOY', 'store_nbr', 'family', 'onpromotion']
features = ['DOY', 'family', 'onpromotion']
#features = ['family', 'onpromotion', 'sin', 'cos']
target = 'sales'







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

y_train = (y_train - mean_train_y)/ std_train_y
y_test = (y_test - mean_train_y)/ std_train_y

sns.violinplot(data=y_train)
sns.violinplot(data=X_train)


#
forest_para = {'n_estimators':[100], 'max_depth':[4],'min_samples_leaf':[4], 'random_state':[42]}

        
forest_reg = RandomForestRegressor()

# Cross validation
grid_search = GridSearchCV(forest_reg, forest_para, cv=5, scoring='neg_mean_squared_error',
return_train_score=True)


grid_search.fit(X_train, y_train)


print(grid_search.best_params_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)



print("R-squared:", r2 )
print("Mean Squared Error (MSE):", mse)


from matplotlib import pyplot as plt


plt.barh(features, best_model.feature_importances_)










