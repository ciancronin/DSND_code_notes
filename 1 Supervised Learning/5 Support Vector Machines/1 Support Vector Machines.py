#
# @author - Cian Cronin (croninc@google.com)
# @description - 1 Support Vector Machines in sklearn
# @date - 13/08/2018
#

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('C:\\Users\\cicro\\Dropbox\\03 Study\\1 Data Science Nanodegree\\1 Supervised Learning\\5 Support Vector Machines\\data.csv')
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

model = SVC(kernel = 'rbf', gamma = 27).fit(X, y)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
print(model)
print(acc)