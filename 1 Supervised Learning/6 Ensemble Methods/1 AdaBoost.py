#
# @author - Cian Cronin (croninc@google.com)
# @description - 1 AdaBoost in sklearn (for reference purposes)
# @date - 14/08/2018
#

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(x_train, y_train)
model.predict(x_test)
y_pred = model.predict(X)

#Don't forget to import the model you want to use for the weak learners
from sklearn.tree import DecisionTreeClassifier
model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)

#Next section is on spam classifying notebook

#Base Model run
# Import our libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Read in our dataset
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])

# Fix our response value
df['label'] = df.label.map({'ham':0, 'spam':1})

# Split our dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

# Instantiate our model
naive_bayes = MultinomialNB()

# Fit our model to the training data
naive_bayes.fit(training_data, y_train)

# Predict on the test data
predictions = naive_bayes.predict(testing_data)

# Score our model
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))

#Importing the functions for the ensembles
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate a BaggingClassifier with:
# 200 weak learners (n_estimators) and everything else as default values
bagging_model = BaggingClassifier(base_estimator = naive_bayes, n_estimators=200)


# Instantiate a RandomForestClassifier with:
# 200 weak learners (n_estimators) and everything else as default values
randomforest_model = RandomForestClassifier(base_estimator = naive_bayes, n_estimators=200)

# Instantiate an a AdaBoostClassifier with:
# With 300 weak learners (n_estimators) and a learning_rate of 0.2
adaboost_model = AdaBoostClassifier(base_estimator = naive_bayes, n_estimators=300)

# Fit your BaggingClassifier to the training data
bagging_model.fit(training_data, y_train)

# Fit your RandomForestClassifier to the training data
randomforest_model.fit(training_data, y_train)

# Fit your AdaBoostClassifier to the training data
adaboost_model.fit(training_data, y_train)

# Predict using BaggingClassifier on the test data
bag_pred = bagging_model.predict(testing_data)

# Predict using RandomForestClassifier on the test data
forest_pred = randomforest_model.predict(testing_data)

# Predict using AdaBoostClassifier on the test data
ada_pred = adaboost_model.predict(testing_data)

#Function to print accuracy scores
def print_metrics(y_true, preds, model_name=None):
    '''
    INPUT:
    y_true - the y values that are actually true in the dataset (numpy array or pandas series)
    preds - the predictions for those values from some model (numpy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to add it to the print statements 
    
    OUTPUT:
    None - prints the accuracy, precision, recall, and F1 score
    '''
    if model_name == None:
        print('Accuracy score: ', format(accuracy_score(y_true, preds)))
        print('Precision score: ', format(precision_score(y_true, preds)))
        print('Recall score: ', format(recall_score(y_true, preds)))
        print('F1 score: ', format(f1_score(y_true, preds)))
        print('\n\n')
    
    else:
        print('Accuracy score for ' + model_name + ' :' , format(accuracy_score(y_true, preds)))
        print('Precision score ' + model_name + ' :', format(precision_score(y_true, preds)))
        print('Recall score ' + model_name + ' :', format(recall_score(y_true, preds)))
        print('F1 score ' + model_name + ' :', format(f1_score(y_true, preds)))
        print('\n\n')

# Print Bagging scores
print_metrics(y_test, bag_pred, model_name = 'bagging_model')

# Print Random Forest scores
print_metrics(y_test, forest_pred, model_name = 'randomforest_model')

# Print AdaBoost scores
print_metrics(y_test, ada_pred, model_name = 'adaboost_model')

# Naive Bayes Classifier scores
print_metrics(y_test, predictions, model_name = 'naivebayes_model')
