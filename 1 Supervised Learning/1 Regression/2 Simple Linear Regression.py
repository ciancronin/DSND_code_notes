#
# @author - Cian Cronin (croninc@google.com)
# @description - 2 Simple Linear Regression Scikit-Learn Implementation
# @date - 04/08/2018
#

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

bmi_life_data = pd.read_csv('/home/cian/Documents/DSND_code_notes/1 Supervised Learning/1 Regression/bmi_and_life_expectancy.csv')

bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data['Life expectancy'])

laos_life_exp = bmi_life_model.predict(np.array[21.07931].reshape(-1, 1))
print(laos_life_exp)
