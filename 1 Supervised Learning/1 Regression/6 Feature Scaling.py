#
# @author - Cian Cronin (croninc@google.com)
# @description - 6 Feature Scaling
# @date - 12/08/2018
#

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

train_data = pd.read_csv("/Users/croninc/Dropbox/03 Study/1 Data Science Nanodegree/1 Regression/data_fs.csv")
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

# TODO: Create the standardization scaling object.
scaler = StandardScaler()

# TODO: Fit the standardization parameters and scale the data.
X_scaled = scaler.fit_transform(X)

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso().fit(X_scaled, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)