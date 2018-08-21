#
# @author - Cian Cronin (croninc@google.com)
# @description - 3 Putting it all together
# @date - 19/08/2018
#

#Exploratory Analysis Notes:
# Import our libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
sns.set(style="ticks")

import check_file as ch

%matplotlib inline

# Read in our dataset
diabetes = pd.read_csv('diabetes.csv')

# Take a look at the first few rows of the dataset
diabetes.head()

# Getting the proportion of Outcome = 1 from total dataset
unique, counts = np.unique(diabetes['Outcome'], return_counts = True)
counts[1]/np.sum(counts)

#Number of missing datapoints
diabetes.describe()
print(diabetes['Pregnancies'].isnull().sum())
print(diabetes['Glucose'].isnull().sum())
print(diabetes['BloodPressure'].isnull().sum())
print(diabetes['SkinThickness'].isnull().sum())
print(diabetes['Insulin'].isnull().sum())
print(diabetes['BMI'].isnull().sum())
print(diabetes['DiabetesPedigreeFunction'].isnull().sum())
print(diabetes['Age'].isnull().sum())
print(diabetes['Outcome'].isnull().sum())

# Plotting the histograms of each column

###### From the solution notebook (pairplotting)
import seaborn as sns
sns.set(style="ticks")
sns.pairplot(diabetes, hue="Outcome") # F

diabetes.hist()

######

plt.style.use('ggplot')
plt.hist(diabetes['Pregnancies']) # Skewed left
plt.hist(diabetes['Glucose']) # Fairly symmetric
plt.hist(diabetes['BloodPressure']) # Fairly symmetric
plt.hist(diabetes['SkinThickness']) # Skewed left
plt.hist(diabetes['Insulin']) # Heavily skewed left
plt.hist(diabetes['BMI']) # Very symmpetric
plt.hist(diabetes['DiabetesPedigreeFunction']) # Heavily skewed left
plt.hist(diabetes['Age']) # Heavily skewed left

#Calculating the correlations

###### From the solution notebook (correlation plotting)
import seaborn as sns
sns.set(style="ticks")
sns.heatmap(diabetes.corr(), annot=True, cmap="YlGnBu");
######

np.corrcoef(diabetes['Pregnancies'], diabetes['Outcome']) # .22
np.corrcoef(diabetes['Glucose'], diabetes['Outcome']) # .47
np.corrcoef(diabetes['BloodPressure'], diabetes['Outcome']) # .07
np.corrcoef(diabetes['SkinThickness'], diabetes['Outcome']) # .07
np.corrcoef(diabetes['Insulin'], diabetes['Outcome']) # .13
np.corrcoef(diabetes['BMI'], diabetes['Outcome']) # .29
np.corrcoef(diabetes['DiabetesPedigreeFunction'], diabetes['Outcome']) # .17
np.corrcoef(diabetes['Age'], diabetes['Outcome']) # .23

#Now putting the above info into action when creating ML models
y = diabetes.iloc[:,-1]
X = diabetes.iloc[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#For a RandomForestClassifier
# build a classifier
clf_rf = RandomForestClassifier()

# Set up the hyperparameter search
param_dist = {"max_depth": [3, None],
              "n_estimators": list(range(10, 200)),
              "max_features": list(range(1, X_test.shape[1]+1)),
              "min_samples_split": list(range(2, 11)),
              "min_samples_leaf": list(range(1, 11)),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# Run a randomized search over the hyperparameters
random_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist)

# Fit the model on the training data
random_search.fit(X_train, y_train)

# Make predictions on the test data
rf_preds = random_search.best_estimator_.predict(X_test)

ch.print_metrics(y_test, rf_preds, 'random forest')

# build a classifier for ada boost
ada_model = AdaBoostClassifier()

# Set up the hyperparameter search
# look at  setting up your search for n_estimators, learning_rate
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
param_dist = {"n_estimators": list(range(10, 200)),
              "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1]}

# Run a randomized search over the hyperparameters
random_search = RandomizedSearchCV(ada_model, param_distributions=param_dist)

# Fit the model on the training data
random_search.fit(X_train, y_train)

# Make predictions on the test data
ada_preds = random_search.best_estimator_.predict(X_test)

# Return your metrics on test data
ch.print_metrics(y_test, ada_preds, 'adaboost')

# build a classifier for support vector machines
svm_model = SVC()

# Set up the hyperparameter search
# look at setting up your search for C (recommend 0-10 range), 
# kernel, and degree
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
param_dist = {"C": [0.1, 0.5, 1, 3, 5],
              "kernel": ['linear','rbf']
             }


# Run a randomized search over the hyperparameters
random_search = RandomizedSearchCV(svm_model, param_distributions=param_dist)

# Fit the model on the training data
random_search.fit(X_train, y_train)

# Make predictions on the test data
svc_preds = random_search.best_estimator_.predict(X_test)

# Return your metrics on test data
ch.print_metrics(y_test, svc_preds, 'svc')