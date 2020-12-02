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

import time
start_time = time.time()

df_raw = pd.read_csv("final_df.csv", index_col=0, parse_dates=True)

# Copy raw file, filter down years and columns
df = df_raw.copy()
df = df.loc['2018-01-01':'2019-12-31']
df = df[['dewpoint', 'rel_humidity', 'temperature', 'wind_direction', 'wind_speed',
          'Fuel_Price', 'Wind_MW', 'Solar_MW', 'Demand_DLAP_MW', 'Demand_MW', 'Year', 
          'Month', 'Day', 'Hour', 'Weekday', 'Weekend', 'LMP_Price_Per_MWh']]


# Feature creation
# Create wind vectors
df['wind_x'] = df['wind_speed'] * np.cos(np.deg2rad(df['wind_direction']))
df['wind_y'] = df['wind_speed'] * np.sin(np.deg2rad(df['wind_direction']))

# Create two weeks worth of lagged variables 168
for i in range (1, (24*7)):
    df['LMP_Price_Per_MWh -' + str(i) + 'h'] = df['LMP_Price_Per_MWh'].shift(i)

# Create 48 hour lagged CAISO data 48
for i in range (1, 24):
    for item in ['Fuel_Price', 'Wind_MW', 'Solar_MW', 'Demand_DLAP_MW', 'Demand_MW']:
        df[item + ' -' + str(i) + 'h'] = df[item].shift(i)

# Create rolling averages
for item in ['Wind_MW', 'Solar_MW', 'Demand_DLAP_MW', 'Demand_MW', 'LMP_Price_Per_MWh', 'temperature']:
    if item in ['Solar_MW', 'Wind_MW']:
        df[item + ' 12 roll avg'] = df[item].rolling(window=12).mean()
    else:
        df[item + ' 4 roll avg'] = df[item].rolling(window=4).mean()

# Target DataFrame
# Create future y's for 
target_df = pd.DataFrame(data = df['LMP_Price_Per_MWh'], index = df.index)
for i in range (1, 24):
    target_df['LMP_Price_Per_MWh +' + str(i) + 'h'] = target_df['LMP_Price_Per_MWh'].shift(-i)
target_df = target_df.drop(labels = ['LMP_Price_Per_MWh'], axis = 1)


# Random split
X_train, X_test, y_train, y_test = train_test_split(df, target_df, test_size=0.25, random_state=1)


# Pre-process Train
ss = pp.StandardScaler()
X_train = pd.DataFrame(ss.fit_transform(X_train), index=y_train.index)

# Drop NANs after to avoid lagged variables losing values and skewing the scaling
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

y_train = y_train.dropna()
X_train = X_train.loc[y_train.index]


# MLP Regressor
regr = MLPRegressor(hidden_layer_sizes = (30, 40, 50, 60, 70), activation = 'relu', solver = 'adam', alpha = 0.05, max_iter=300)
regr.fit(X_train, y_train)
print('R^2:', regr.score(X_train, y_train))

print('Time to finish: ', (time.time()-start_time)/60)

