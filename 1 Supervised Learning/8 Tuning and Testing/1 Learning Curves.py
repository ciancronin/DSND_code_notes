#
# @author - Cian Cronin (croninc@google.com)
# @description - 1 Learning Curves
# @date - 17/08/2018
#

# Import, read, and split data
import pandas as pd
data = pd.read_csv('C:\\Users\\cicro\\Documents\\GitHub\\datasciencenanodegree\\1 Supervised Learning\\8 Tuning and Testing\\data.csv')
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Fix random seed
np.random.seed(55)

### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

### Logistic Regression
estimator = LogisticRegression()

### Decision Tree
#estimator = GradientBoostingClassifier()

### Support Vector Machine
#estimator = SVC(kernel='rbf', gamma=1000)