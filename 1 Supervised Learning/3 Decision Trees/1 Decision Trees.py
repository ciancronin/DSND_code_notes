#
# @author - Cian Cronin (croninc@google.com)
# @description - 1 Decision Trees
# @date - 12/08/2018
#

# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = pd.read_csv('C:\\Users\\cicro\\Documents\\GitHub\\datasciencenanodegree\\1 Supervised Learning\\3 Decision Trees\\data.csv')
# Assign the features to the variable X, and the labels to the variable y. 
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier().fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)
print(acc)