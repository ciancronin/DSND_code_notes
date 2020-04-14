"""Snippets for GradientBoostingRegressor"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import datetime as dt
import math
from itertools import product
import numpy as np
import time
import utils as ut
import sys


def find_best_gbm_model(X_train, y_train, X_valid, y_valid, parameters):
  """Train a GradientBoostedRegressor Model."""

  # Train based on parameters dict
  grid_model_list = {}
  for i, params in enumerate(parameters_flattened):
    model = GradientBoostingRegressor(**params)

    model.fit(X_train, y_train)

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
  data = yf.download("GOOG AAPL", period='3y')

  goog_adj_close = pd.DataFrame(data['Adj Close']['GOOG'].values,
                                columns=['amount'],
                                index=data['Adj Close']['GOOG'].index)

  y_tr, y_val, y_tst, X_tr, X_val, X_tst = ut.preprocess_time_series_and_split(
      goog_adj_close)

  parameters = {
      'learning_rate': [0.1, 0.05],
      'max_depth': [4],  # [4, 6, 8],
      'n_estimators': [60],  # [60, 80, 100, 120],
      'subsample': [0.8],
      'loss': ['ls'],  # Least-squares
      'criterion': ['mse']
  }

  parameters_flattened = [dict(zip(parameters, v)) for v in product(*parameters.values())]
  print('{} parameter combinations to train'.format(len(parameters_flattened)))

  model = find_best_gbm_model(X_tr, y_tr, X_val, y_val, parameters)

  model_preds = model.predict(X_tst)

  print(ut.smape(y_tst['amount'], model_preds))

  fig, ax = plt.subplots(figsize=(20, 10))
  ax.plot(y_tst.values, label='Actuals')
  ax.plot(model_preds, label='Predictions')
  ax.legend()
  plt.show()

