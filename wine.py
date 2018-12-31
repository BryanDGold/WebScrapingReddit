import pandas as pd 
import numpy as np
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt


wine_file_path = '/Users/BryanGoldberg/Desktop/Personal Projects/winequality.csv'
wine = pd.read_csv(wine_file_path)
wine.describe()
#Viewing summary statistics of our dataset, with statistics such as count, mean, standard deviation, minimum, 25th, 50th, and 75th percentiles, and maximum values.

plt.hist(wine)

wine = pd.get_dummies(wine)
wine.iloc[:,5:].head(5)

labels = np.array(wine['quality'])
wine = wine.drop('quality', axis = 1)
wine_list = list(wine.columns)
wine = np.array(wine)
#Features and Targets and Convert Data to Arrays

from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(wine, labels, test_size = .25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape', test_features.shape)
print('Testing Labels Shape', test_labels.shape)
#Splitting Data into Test and Training Sets

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'quality points')

mape = 100 * (errors / test_labels)
#Calculate mean absolute error percentage

accuracy = 100 - np.mean(mape)
#Calculate and Display Accuracy

print('Accuracy:', round(accuracy, 2), '%')

from sklearn.tree import export_graphviz 
import pydot

tree = rf.estimators_[5]

export_graphviz(tree, out_file = 'tree.dot', feature_names = wine_list, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')

graph.write_png(tree.png)

importances = list(rf.feature_importances_)

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(wine_list, importances)]

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#Figuring out which variables are the most important when building our model.

rf_most_important = RandomForestRegressor(n_estimators = 1000, random_state = 42)
#Starting to build another random forest with the two most important variables that we found in the last section.
important_indices = [wine_list.index('alcohol'), wine_list.index('sulphates')]
#Extracting the two most important features.
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

rf_most_important.fit(train_important, train_labels)
#Train the random forest

predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
#Making predictions and determing the error

print('Mean Absolute Error', round(np.mean(errors), 2), 'quality points')
mape = np.mean(100 * (errors / test_labels))
print('Accuracy:', round(accuracy, 2), '%')
#Displaying Performance Metrics

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

x_values = list(range(len(importances)))

plt.bar(x_values, importances, orientation = 'vertical')
#Making a bar chart of the different feature importancs.

plt.xticks(x_values, wine_list, rotation = 'vertical')
#Tick labels for x-axis

plt.ylabel('Importance'); plt.xlabel('Variable')
plt.title('Variable Importances')

