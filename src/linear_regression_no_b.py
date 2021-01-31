# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

#%%
df_train_phantom = pd.read_csv('data/phantom_training_no_b.csv')
df_test_phantom = pd.read_csv('data/phantom_testing_no_b.csv')

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

df_train_ellastic = pd.read_csv('data/ellastic_training_no_b.csv')
df_test_ellastic = pd.read_csv('data/ellastic_testing_no_b.csv')

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

df_train_collision_when = pd.read_csv('data/collision_when_training_no_b.csv')
df_test_collision_when = pd.read_csv('data/collision_when_testing_no_b.csv')

x_collision_when = df_train_collision_when[['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b']]

y_collision_when = df_train_collision_when['collision time']

x_test_collision_when = df_test_collision_when[['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b']]

y_test_collision_when = df_test_collision_when['collision time']

linear_regression_collision_when = LinearRegression()
linear_regression_collision_when.fit(x_collision_when, y_collision_when)

y_pred_collision_when = linear_regression_collision_when.predict(x_test_collision_when)

df_comp_collision_when = pd.DataFrame({'Actual': y_test_collision_when, 'Predicted': y_pred_collision_when})


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_collision_when, y_pred_collision_when))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_collision_when, y_pred_collision_when)))


df_comp_collision_when.sort_values(by = ['Actual'], inplace = True)
df_comp_collision_when = df_comp_collision_when.reset_index(drop=True)

plt.scatter(x = df_comp_collision_when.index, y = df_comp_collision_when.Predicted, color='blue')         
plt.scatter(x = df_comp_collision_when.index, y = df_comp_collision_when.Actual, color='black')
plt.show()

#%%



df_train_collision_if = pd.read_csv('data/collision_if_training.csv')
df_test_collision_if = pd.read_csv('data/collision_if_testing.csv')

x_collision_if = df_train_collision_if[['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b']]

y_collision_if = df_train_collision_if['if collided']

x_test_collision_if = df_test_collision_if[['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b']]

y_test_collision_if = df_test_collision_if['if collided']

logistic_regression_collision_if = LogisticRegression()
logistic_regression_collision_if.fit(x_collision_if, y_collision_if)

y_pred_collision_if = logistic_regression_collision_if.predict(x_test_collision_if)

score = logistic_regression_collision_if.score(x_test_collision_if, y_test_collision_if)
print(score)
#%%

df_comp_collision_if = pd.DataFrame({'Actual': y_test_collision_if, 'Predicted': y_pred_collision_if})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_collision_if, y_pred_collision_if))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_collision_if, y_pred_collision_if)))


df_comp_collision_if.sort_values(by = ['Actual'], inplace = True)
df_comp_collision_if = df_comp_collision_if.reset_index(drop=True)

plt.scatter(x = df_comp_collision_if.index, y = df_comp_collision_if.Predicted, color='blue')         
plt.scatter(x = df_comp_collision_if.index, y = df_comp_collision_if.Actual, color='black')
plt.show()

#%%

df_train_collision_if_dist = pd.read_csv('data/collision_if_dist_training_no_b.csv')
df_test_collision_if_dist = pd.read_csv('data/collision_if_dist_testing_no_b.csv')

x_collision_if_dist = df_train_collision_if_dist[[
        'initial velocity of a',
        'initial velocity of b']]

y_collision_if_dist = df_train_collision_if_dist['if collided']

x_test_collision_if_dist = df_test_collision_if_dist[[
        'initial velocity of a',
        'initial velocity of b']]

y_test_collision_if_dist = df_test_collision_if_dist['if collided']

y_collision_if_dist.loc[y_collision_if_dist > 0] = 1
y_test_collision_if_dist.loc[y_test_collision_if_dist > 0] = 1


logistic_regression_collision_if_dist = LogisticRegression()
logistic_regression_collision_if_dist.fit(x_collision_if_dist, y_collision_if_dist)

y_pred_collision_if_dist = logistic_regression_collision_if_dist.predict(x_test_collision_if_dist)

score = logistic_regression_collision_if_dist.score(x_test_collision_if_dist, y_test_collision_if_dist)
print(score)

cnf_matrix = metrics.confusion_matrix(y_test_collision_if_dist, y_pred_collision_if_dist)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#%%

df_train_collision_if = pd.read_csv('data/collision_if_training.csv')
df_test_collision_if = pd.read_csv('data/collision_if_testing.csv')

x_collision_if_dist = df_train_collision_if[['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b']]

y_collision_if_dist = df_train_collision_if['if collided']

x_test_collision_if_dist = df_test_collision_if[['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b']]

y_test_collision_if_dist = df_test_collision_if['if collided']

y_collision_if_dist.loc[y_collision_if_dist > 0] = 1
y_test_collision_if_dist.loc[y_test_collision_if_dist > 0] = 1


logistic_regression_collision_if_dist = LogisticRegression()
logistic_regression_collision_if_dist.fit(x_collision_if_dist, y_collision_if_dist)

y_pred_collision_if_dist = logistic_regression_collision_if_dist.predict(x_test_collision_if_dist)

score = logistic_regression_collision_if_dist.score(x_test_collision_if_dist, y_test_collision_if_dist)
print(score)

cnf_matrix = metrics.confusion_matrix(y_test_collision_if_dist, y_pred_collision_if_dist)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#%%


