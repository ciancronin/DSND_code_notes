"""Utils Snippet Class."""
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import math
import sys

def difference(dataset, interval=1):
  """Return a differenced Series of dataset."""
  diff = list()
  for i in range(interval, len(dataset)):
    value = dataset[i] - dataset[i - interval]
    diff.append(value)
  return pd.Series(diff)

def inverse_difference(last_ob, forecast):
  """Function to inverse a differenced forecast."""
  inverted = list()
  inverted.append(forecast[0] + last_ob)
  for i in range(1, len(forecast)):
    inverted.append(forecast[i] + inverted[i-1])
  return inverted

def scale_data(series, scaler_range=(-1, 1), interval=1):
  """Scale Series between range."""
  scaler = MinMaxScaler(feature_range=scaler_range)
  scaled_values = scaler.fit_transform(series)
  scaled_values = scaled_values.reshape(len(scaled_values), 1)

  return scaler, scaled_values

def inverse_transform(series, forecasts, scaler, n_test):
  """For inverse difference and scaler from forecasts."""
  inverted = list()
  for i in range(len(forecasts)):
    # create array from forecast
    forecast = array(forecasts[i])
    forecast = forecast.reshape(1, len(forecast))
    # invert scaling
    inv_scale = scaler.inverse_transform(forecast)
    inv_scale = inv_scale[0, :]
    # invert differencing
    index = len(series) - n_test + i - 1
    last_ob = series.values[index]
    inv_diff = inverse_difference(last_ob, inv_scale)
    # store
    inverted.append(inv_diff)
  return inverted

def smape(A, F):
  """Return the SMAPE evaluation metric for a set of predictions."""
  A = np.asarray(A)
  F = np.asarray(F)
  return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def huber_approx_obj(preds, dtrain):
  """Huber Loss Approximation Function."""
  d = preds - dtrain.get_labels()
  h = 1
  scale = 1+ (d / h) ** 2
  scale_sqrt = np.sqrt(scale)
  grad = d / scale_sqrt
  hess = 1 / scale / scale_sqrt
  return grad, hess

def preprocess_time_series_and_split(df, train_pct=0.7, valid_pct=0.15):
  """Breakout of time series into various features for modelling

  TODO(add more documentation)
  """

  months = []
  weeks = []
  day_of_week = []
  day_of_year = []
  years = []
  amount_ly = []
  amount_tya = []

  dummies = pd.DataFrame(index=df.index)

  for day in df.index:
    # Creating time based features
    months.append(day.strftime('%m'))
    weeks.append(day.strftime('%W'))
    day_of_week.append(day.strftime('%w'))
    day_of_year.append(day.strftime('%j'))
    years.append(day.strftime('%y'))

    # Getting amount based features
    day_ly = day - dt.timedelta(days=364)
    day_tya = day - dt.timedelta(days=728)
    amount_ly.append(df[df.index == day_ly]['amount'].values)
    amount_tya.append(df[df.index == day_tya]['amount'].values)

  dummies['month'] = months
  dummies['week'] = weeks
  dummies['day_of_week'] = day_of_week
  dummies['day_of_year'] = day_of_year
  dummies['year'] = years

  month_dummies = pd.get_dummies(dummies['month'], prefix='month')
  month_dummies = pd.DataFrame(month_dummies,
                               columns=month_dummies.columns).\
    set_index(dummies.index)

  week_dummies = pd.get_dummies(dummies['week'], prefix='week')
  week_dummies = pd.DataFrame(week_dummies,
                              columns=week_dummies.columns).\
    set_index(dummies.index)

  day_of_week_dummies = pd.get_dummies(
      dummies['day_of_week'], prefix='day_of_week')
  day_of_week_dummies = pd.DataFrame(day_of_week_dummies,
                                     columns=day_of_week_dummies.columns).\
    set_index(dummies.index)

  day_of_year_dummies = pd.get_dummies(
      dummies['day_of_year'], prefix='day_of_year')
  day_of_year_dummies = pd.DataFrame(day_of_year_dummies,
                                     columns=day_of_year_dummies.columns).\
    set_index(dummies.index)

  year_dummies = pd.get_dummies(dummies['year'], prefix='year')
  year_dummies = pd.DataFrame(year_dummies,
                              columns=year_dummies.columns).\
    set_index(dummies.index)

  X_st = df.join([month_dummies, week_dummies,
                  day_of_week_dummies, day_of_year_dummies, year_dummies])

  amount_ly = pd.DataFrame(
    amount_ly, columns=['amount_ly']).set_index(df.index).fillna(0.0)
  amount_tya = pd.DataFrame(
    amount_tya, columns=['amount_tya']).set_index(df.index).fillna(0.0)

  X_st = X_st.merge(amount_ly, left_index=True, right_index=True)
  X_st = X_st.merge(amount_tya, left_index=True, right_index=True)

  y = pd.DataFrame(X_st['amount'], columns=['amount'])

  X = X_st.drop('amount', axis=1).fillna(0)

  train_end = math.floor(len(y) * train_pct)
  valid_end = math.floor(len(y) * valid_pct) + train_end

  y_train = y.iloc[0:train_end, :]
  y_valid = y.iloc[train_end:valid_end, :]
  y_test = y.iloc[valid_end:, :]

  X_train = X.iloc[0:train_end, :]
  X_valid = X.iloc[train_end:valid_end, :]
  X_test = X.iloc[valid_end:, :]

  return y_train, y_valid, y_test, X_train, X_valid, X_test
