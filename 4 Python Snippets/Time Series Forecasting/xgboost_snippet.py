"""Snippets for XGBoostRegressor"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import datetime as dt
import math
from itertools import product
import numpy as np
import time
import utils as ut
import sys

def error_plot(model, figsize=(5, 5)):
  results = model.evals_result_
  epochs = len(results['validation_0']['mae'])
  x_axis = range(0, epochs)

  # Plot log loss
  fig, ax = plt.subplots(figsize=figsize)
  ax.plot(x_axis, results['validation_0']['mae'], label='Train')
  ax.plot(x_axis, results['validation_1']['mae'], label='Test')
  ax.legend()
  plt.ylabel('MAE')
  plt.title('XGBoost MAE')
  plt.show()

def find_best_xgb_model(X_train, y_train, X_valid, y_valid, parameters):
  """Train a GradientBoostedRegressor Model."""

  # Train based on parameters dict
  grid_model_list = {}
  for i, params in enumerate(parameters_flattened):
    model = xgb.XGBRegressor(**params)
    eval_set = [(X_train, y_train),
                (X_valid, y_valid)]

    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    # Store the model
    grid_model_list[i] = model

  # Store train and valid evaluation results
  model_result = []
  for i, m in grid_model_list.items():

    d = {}
    d['id'] = i

    pred = pd.Series(m.predict(X_train)).clip(lower=0)
    d['train'] = mean_absolute_error(pred, y_train)

    pred = pd.Series(m.predict(X_valid)).clip(lower=0)
    d['valid'] = mean_absolute_error(pred, y_valid)

    model_result.append(d)

  model_result_df = pd.DataFrame(model_result).sort_values('valid')

  return grid_model_list[model_result_df.iloc[0].name]
  

if __name__ == '__main__':
  data = yf.download("GOOG AAPL", period='5y')

  goog_adj_close = pd.DataFrame(data['Adj Close']['GOOG'].values,
                                columns=['amount'],
                                index=data['Adj Close']['GOOG'].index)

  y_tr, y_val, y_tst, X_tr, X_val, X_tst = ut.preprocess_time_series_and_split(
      goog_adj_close, train_pct=0.5, valid_pct=0.25)

  parameters = {
      'learning_rate': [0.1, 0.05],
      'max_depth': [4, 6, 8],
      'n_estimators': [60, 80, 100, 120],
      'subsample': [0.8],
      'colsample_bytree': [0.8],
      'lambda': [0.5],  # L2 Reg. Higher is more conservative
      'alpha': [0.5],  # L1 Reg. Higher is more conservative
      'obj': [ut.huber_approx_obj],
      'booster': ['gbtree'],  # Least-squares
      'eval_metric': ['mae']
  }

  parameters_flattened = [dict(zip(parameters, v)) for v in product(*parameters.values())]
  print('{} parameter combinations to train'.format(len(parameters_flattened)))

  model = find_best_xgb_model(X_tr, y_tr, X_val, y_val, parameters)

  error_plot(model)

  xgb.plot_importance(model, max_num_features=10)
  plt.show()

  model_preds = model.predict(X_tst)

  print('MAE Test: {}'.format(mean_absolute_error(y_tst.values, model_preds)))
  print('SMAPE Test: {}'.format(ut.smape(y_tst['amount'].values, model_preds)))

  fig, ax = plt.subplots(figsize=(20, 10))
  ax.plot(y_tst.values, label='Actuals')
  ax.plot(model_preds, label='Predictions')
  ax.legend()
  plt.show()


