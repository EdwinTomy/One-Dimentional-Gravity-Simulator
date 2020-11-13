# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
df_train_phantom = pd.read_csv('data/phantom_training.csv')
df_test_phantom = pd.read_csv('data/phantom_testing.csv')

x_phantom = df_train_phantom[['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b']]

y_phantom = df_train_phantom['final position of a']

x_test_phantom = df_test_phantom[['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b']]

y_test_phantom = df_test_phantom['final position of a']

linear_regression_phantom = LinearRegression()
linear_regression_phantom.fit(x_phantom, y_phantom)

y_pred_phantom = linear_regression_phantom.predict(x_test_phantom)

df_comp_phantom = pd.DataFrame({'Actual': y_test_phantom, 'Predicted': y_pred_phantom})


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_phantom, y_pred_phantom))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_phantom, y_pred_phantom)))


df_comp_phantom.sort_values(by = ['Actual'], inplace = True)
df_comp_phantom = df_comp_phantom.reset_index(drop=True)

plt.scatter(x = df_comp_phantom.index, y = df_comp_phantom.Predicted, color='blue')         
plt.scatter(x = df_comp_phantom.index, y = df_comp_phantom.Actual, color='black')
plt.show()


#%%

df_train_ellastic = pd.read_csv('data/ellastic_training.csv')
df_test_ellastic = pd.read_csv('data/ellastic_testing.csv')

x_ellastic = df_train_ellastic[['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b']]

y_ellastic = df_train_ellastic['final position of a']

x_test_ellastic = df_test_ellastic[['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b']]

y_test_ellastic = df_test_ellastic['final position of a']

linear_regression_ellastic = LinearRegression()
linear_regression_ellastic.fit(x_ellastic, y_ellastic)

y_pred_ellastic = linear_regression_ellastic.predict(x_test_ellastic)

df_comp_ellastic = pd.DataFrame({'Actual': y_test_ellastic, 'Predicted': y_pred_ellastic})


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_ellastic, y_pred_ellastic))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_ellastic, y_pred_ellastic)))


df_comp_ellastic.sort_values(by = ['Actual'], inplace = True)
df_comp_ellastic = df_comp_ellastic.reset_index(drop=True)

plt.scatter(x = df_comp_ellastic.index, y = df_comp_ellastic.Predicted, color='blue')         
plt.scatter(x = df_comp_ellastic.index, y = df_comp_ellastic.Actual, color='black')
plt.show()

#%%