# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:57:19 2020

@author: bnebe
"""

import pandas as pd
import numpy as np

import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score


# Read csv as raw df
df_raw = pd.read_csv("final_df.csv", index_col=0, parse_dates=True)

# Create df to use as training set
df_train = df_raw.copy()

df_train = df_train.loc['2018-01-01':'2019-9-30']
df_train = df_train[['dewpoint', 'rel_humidity', 'temperature', 'wind_direction', 'wind_speed',
          'Fuel_Price', 'Wind_MW', 'Solar_MW', 'Demand_DLAP_MW', 'Demand_MW', 'Year', 
          'Month', 'Day', 'Hour', 'Weekday', 'Weekend', 'LMP_Price_Per_MWh']]

# Create wind vectors
df_train['wind_x'] = df_train['wind_speed'] * np.cos(np.deg2rad(df_train['wind_direction']))
df_train['wind_y'] = df_train['wind_speed'] * np.sin(np.deg2rad(df_train['wind_direction']))


# Create two weeks worth of lagged variables
for i in range (1, (24*7)):
    df_train['LMP_Price_Per_MWh -' + str(i) + 'h'] = df_train['LMP_Price_Per_MWh'].shift(i)

# for i in range ((24*6), (24*7)):
#     df_train['LMP_Price_Per_MWh -' + str(i) + 'h'] = df_train['LMP_Price_Per_MWh'].shift(i)

# Create 48 hour lagged CAISO data
for i in range (1, 48):
    for item in ['Fuel_Price', 'Wind_MW', 'Solar_MW', 'Demand_DLAP_MW', 'Demand_MW']:
        df_train[item + ' -' + str(i) + 'h'] = df_train[item].shift(i)

# Create future y's for 
target_df = pd.DataFrame(data = df_train['LMP_Price_Per_MWh'], index = df_train.index)
for i in range (1, 24):
    target_df['LMP_Price_Per_MWh +' + str(i) + 'h'] = target_df['LMP_Price_Per_MWh'].shift(-i)
target_df = target_df.drop(labels = ['LMP_Price_Per_MWh'], axis = 1)

# Create rolling averages
for item in ['Wind_MW', 'Solar_MW', 'Demand_DLAP_MW', 'Demand_MW', 'LMP_Price_Per_MWh', 'temperature']:
    if item in ['Solar_MW', 'Wind_MW']:
        df_train[item + ' 12 roll avg'] = df_train[item].rolling(window=12).mean()
    else:
        df_train[item + ' 4 roll avg'] = df_train[item].rolling(window=4).mean()


ss = pp.StandardScaler()
df_train = pd.DataFrame(ss.fit_transform(df_train), index=df_train.index)

df_train = df_train.dropna()
target_df = target_df.loc[df_train.index]

target_df = target_df.dropna()
df_train = df_train.loc[target_df.index]


model = MLPRegressor(hidden_layer_sizes = (25, 400, 400, 25), activation = 'relu', solver = 'adam', alpha = 0.05, max_iter=300)
model.fit(df_train, target_df)


# Create test dataframe
df_test = df_raw.copy()
df_test = df_test.loc['2019-10-01':'2019-12-31']

df_test = df_test[['dewpoint', 'rel_humidity', 'temperature', 'wind_direction', 'wind_speed',
          'Fuel_Price', 'Wind_MW', 'Solar_MW', 'Demand_DLAP_MW', 'Demand_MW', 'Year', 
          'Month', 'Day', 'Hour', 'Weekday', 'Weekend', 'LMP_Price_Per_MWh']]

# Create wind vectors
df_test['wind_x'] = df_test['wind_speed'] * np.cos(np.deg2rad(df_test['wind_direction']))
df_test['wind_y'] = df_test['wind_speed'] * np.sin(np.deg2rad(df_test['wind_direction']))


# Create two weeks worth of lagged variables
for i in range (1, (24*7)):
    df_test['LMP_Price_Per_MWh -' + str(i) + 'h'] = df_test['LMP_Price_Per_MWh'].shift(i)
    
# for i in range ((24*6), (24*7)):
#     df_test['LMP_Price_Per_MWh -' + str(i) + 'h'] = df_test['LMP_Price_Per_MWh'].shift(i)

# Create 48 hour lagged CAISO data
for i in range (1, 48):
    for item in ['Fuel_Price', 'Wind_MW', 'Solar_MW', 'Demand_DLAP_MW', 'Demand_MW']:
        df_test[item + ' -' + str(i) + 'h'] = df_test[item].shift(i)

# Create future y's for 
target_test_df = pd.DataFrame(data = df_test['LMP_Price_Per_MWh'], index = df_test.index)
for i in range (1, 24):
    target_test_df['LMP_Price_Per_MWh +' + str(i) + 'h'] = target_test_df['LMP_Price_Per_MWh'].shift(-i)
target_test_df = target_test_df.drop(labels = ['LMP_Price_Per_MWh'], axis = 1)

# Create rolling averages
for item in ['Wind_MW', 'Solar_MW', 'Demand_DLAP_MW', 'Demand_MW', 'LMP_Price_Per_MWh', 'temperature']:
    if item in ['Solar_MW', 'Wind_MW']:
        df_test[item + ' 12 roll avg'] = df_test[item].rolling(window=12).mean()
    else:
        df_test[item + ' 4 roll avg'] = df_test[item].rolling(window=4).mean()


# DropNAs
columns_list = df_test.columns

df_test = df_test.dropna()
target_test_df = target_test_df.loc[df_test.index]

target_test_df = target_test_df.dropna()
df_test = df_test.loc[target_test_df.index]

df_test = pd.DataFrame(ss.transform(df_test), index=df_test.index)

pred = model.predict(df_test)
pred = pd.DataFrame(pred)



import matplotlib.pyplot as plt

for i in range(0,23):
    y_true, y_pred = np.array(target_test_df.iloc[:,i]), np.array(pred.iloc[:,i])
    residuals = (y_true- y_pred)
    fig, axs = plt.subplots(2, figsize=(15,15))
    
    y_true = pd.DataFrame(data=y_true, index=df_test.index)
    y_pred = pd.DataFrame(data=y_pred, index=df_test.index)
    
    axs[0].plot(y_true)
    axs[0].plot(y_pred, alpha=0.6)
    
    axs[1].plot(residuals)
    plt.show()
    print('MAPE:',(np.mean(np.abs((y_true - y_pred) / y_true)) * 100))




# df_test.reset_index(drop=True, inplace=True)
# pred.reset_index(drop=True, inplace=True)



from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# y_true, y_pred = np.array(target_test_df.iloc[:,0]), np.array(pred.iloc[:,0])
# MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# print('MAPE:', MAPE)
print('Mean Squared Error:', mean_squared_error(target_test_df, pred))
print('R^2:', r2_score(target_test_df, pred))





# # Parameters to hypertune
# param_grid = {'model__max_depth': [200, 250], 'model__criterion': ['mse', 'mae']}

# # Hypertune based on parameters in param_grid
# search = GridSearchCV(pipe, param_grid, cv=2)

# # Fit training set based on 
# search.fit(X_train, y_train)

# print(search.best_estimator_,'\n')
# print(search.best_score_,'\n')
# print(search.best_params_,'\n')
# print(search.cv_results_,'\n')


# # KFold split
# # Need to drop index for it to work
# #======================================================================================================#
# kf = KFold(n_splits=3)

# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X.loc[train_index], X.loc[test_index]
#     y_train, y_test = y.loc[train_index], y.loc[test_index]
# #======================================================================================================#
# scores = cross_val_score(dtr, X=X, y=y, cv=3)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# #======================================================================================================#
