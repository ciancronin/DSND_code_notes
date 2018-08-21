#
# @author - Cian Cronin (croninc@google.com)
# @description - 5 Regularisation
# @date - 12/08/2018
#

# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv("/Users/croninc/Dropbox/03 Study/1 Data Science Nanodegree/1 Regression/data_reg.csv")
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso().fit(X, y)


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)