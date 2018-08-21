#
# @author - Cian Cronin (croninc@google.com)
# @description - 4 Polynomial Regression Scikit-Learn Implementation
# @date - 11/08/2018
#

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('C:\\Users\\cicro\\Dropbox\\03 Study\\1 Data Science Nanodegree\\1 Regression\\data_poly.csv')
X = train_data['Var_X'].values.reshape(-1,1)
y = train_data['Var_Y'].values

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)

# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!