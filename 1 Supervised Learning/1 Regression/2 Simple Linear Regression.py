#
# @author - Cian Cronin (croninc@google.com)
# @description - 2 Simple Linear Regression Scikit-Learn Implementation
# @date - 04/08/2018
#

from sklearn.linear_model import LinearRegression
import pandas as pd

<<<<<<< HEAD
bmi_life_data = pd.read_csv('C:\\Users\\cicro\\Documents\\GitHub\\datasciencenanodegree\\1 Supervised Learning\\1 Regression\\bmi_and_life_expectancy.csv')
=======
bmi_life_data = pd.read_csv('C:\\Users\\cicro\\Dropbox\\3 Study\\'
                            '1 Data Science Nanodegree\\1 Supervised Learning'
                            '\\1 Regression\\bmi_and_life_expectancy.csv')
>>>>>>> f84a9fc73f2fd00369bbec40fcaf53c11d235d37

bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data['Life expectancy'])

laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)
