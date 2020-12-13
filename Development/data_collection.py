# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:00:10 2020

@author: bnebe
"""

import os
import sys

def add_module_path_to_system():
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
        return module_path 

module_path = add_module_path_to_system()

import pandas as pd
import numpy as np

import time
import datetime
import matplotlib.pyplot as plt

from caiso_data_manager import end_day_of_month
from caiso_data_manager import daylight_savings_correction
import data_pull_funcs as dat


start_time = time.time()


# Create date variables
end_year = datetime.date.today().year
end_month = datetime.date.today().month
end_day = end_day_of_month(end_year, end_month)

start_year = end_year-1
start_month = 1
start_day = 1


#Take weather dataframes and rather than adjust times create lines between weather points and use those as datapoints, rather than forcing data point at 4:30 to be at 4
start_date = datetime.date(start_year, start_month, start_day)
end_date = datetime.date(end_year, end_month, end_day) + datetime.timedelta(days=8)


# Create date_range for start date to end date with additional days for forecast data
date_range = pd.date_range(start_date, end_date, freq='H', tz='utc')
date_range = pd.DataFrame(data=None, index=date_range)


# Read CSV of zipcodes for area
zips = pd.read_csv('PGE_2019_Q4_ElectricUsageByZip.csv', usecols=['ZIPCODE'])
zips.drop_duplicates(keep='first', inplace=True)
zips.reset_index(drop=True,inplace=True)

# Convert zipscodes to coordinates and find usable weather stations in those coordinates
points = dat.convert_zips_to_coor(zips)
station_coor = dat.convert_coor_to_stations(points, 2)
sample = dat.iterative_farthest_sample_check(station_coor, 12, check=True)
print('Time to sample points: ', (time.time()-start_time)/60)


# NREL DataFrame
# NREL Meters per second for wind speed (1m/s = 2.23694mph)
# Create NREL dataframe from prior year
NREL_df = pd.DataFrame()
for x in range(end_year - start_year + 1):
    year = start_year + x
    temp_df = dat.NREL_combine_obs(sample, year)
    NREL_df = pd.concat([NREL_df, temp_df], axis=0)

print('Time to create NREL df: ', (time.time()-start_time)/60)


# NOAA DataFrame
#Lists to replace dictionaries/lists with values or remove features all together
replace_list = ['barometricPressure', 'dewpoint', 'elevation', 'heatIndex', 'relativeHumidity', 'seaLevelPressure',
                'temperature', 'visibility', 'windChill', 'windDirection', 'windGust', 'windSpeed']

remove_list = ['@id', '@type', 'maxTemperatureLast24Hours', 'minTemperatureLast24Hours',
                'precipitationLast3Hours', 'precipitationLast6Hours', 'precipitationLastHour', 
                'presentWeather', 'rawMessage', 'textDescription', 'cloudLayers', 'icon', 'station']

# Create NOAA observations dataframe
NOAA_obs_df = dat.NOAA_combine_obs(sample, replace_list, remove_list)
print('Time to create NOAA obs df: ', (time.time()-start_time)/60)



# Create NOAA forecast dataframe
keep_list = ['startTime', 'temperature', 'windSpeed']
NOAA_forc_df = dat.NOAA_combine_forc(sample, keep_list)

# Fill blanks in NOAA forecast df using linear interpolation
forc_sdate = NOAA_obs_df.index.max() + datetime.timedelta(hours=1)
forc_edate = NOAA_forc_df.index.max()


forc_date_range = pd.date_range(forc_sdate, forc_edate, freq='H', tz='utc')
forc_date_range = pd.DataFrame(data=None, index=forc_date_range)
NOAA_forc_df = forc_date_range.merge(NOAA_forc_df, how='left', left_index=True, right_index=True)

NOAA_forc_df = NOAA_forc_df.interpolate(method='linear', axis=0, limit_direction='both')

print('Time to create NOAA forc df: ', (time.time()-start_time)/60)



# CAISO DataFrame
output_folder = "D:/Users/bnebe/Documents/Test_Temp"  # Provide a new location preferably like your desktop even though folder doesn't exist, it will be created 
area_name = ['PGE-TAC']
trading_hub = ['NP15', 'SP15', 'ZP26']
node = "DLAP_PGAE-APND"


# Create CAISO dataframe 
CAISO_df_raw = dat.caiso_data_pull(output_folder, start_year, start_month, start_day, end_year, end_month, end_day, node, area_name, trading_hub)

CAISO_sdate = CAISO_df_raw.index.min()
CAISO_edate = CAISO_df_raw.index.max()

CAISO_date_range = pd.date_range(CAISO_sdate, CAISO_edate, freq='H', tz='utc')
CAISO_date_range = pd.DataFrame(data=None, index=CAISO_date_range)
CAISO_df = CAISO_date_range.merge(CAISO_df_raw, how='left', left_index=True, right_index=True)

# Find Solar Zero Hours
solar_df = CAISO_df_raw['Solar_MW']
solar_df = CAISO_date_range.merge(solar_df, how='left', left_index=True, right_index=True)
hours_dict = dat.create_dict_solar_zero_hours(solar_df, 50)

for index, row in CAISO_df.iterrows():
    cur_month = index.month
    cur_hour = index.hour
    if (pd.isnull(row['Solar_MW'])) and (cur_hour in hours_dict[cur_month]):
        CAISO_df.loc[index, 'Solar_MW'] = 0


linear_columns = ['LMP_Price_Per_MWh', 'Wind_MW', 'Demand_DLAP_MW', 'Demand_MW', 'Solar_MW']
ffill_columns = ['Fuel_Price']

for col in CAISO_df.columns:
    if col in linear_columns:
        CAISO_df[col] = CAISO_df[col].interpolate(method='linear', axis=0, limit_direction='both')
    elif col in ffill_columns:
        CAISO_df[col] = CAISO_df[col].interpolate(method='ffill', axis=0, limit_direction='forward')

print('Time to create CAISO df: ', (time.time()-start_time)/60)



# Filter and rename columns for dataframes to merge
NREL_df = NREL_df[['Dew Point', 'Relative Humidity', 'Temperature', 'Wind Direction', 'Wind Speed', 'Solar Zenith Angle']]
NREL_df.columns = ['dewpoint', 'rel_humidity', 'temperature', 'wind_direction', 'wind_speed', 'SZA']

NOAA_obs_df.columns = ['barometric_pressure', 'dewpoint', 'elevation', 'heat_index', 
                        'rel_humidity', 'sea_level_pressure', 'temperature', 'visibility', 
                        'wind_chill', 'wind_direction', 'wind_gust', 'wind_speed']

NOAA_forc_df = NOAA_forc_df[['temperature', 'windSpeed']]
NOAA_forc_df.columns = ['temperature', 'wind_speed']

CAISO_df = CAISO_df[['LMP_Price_Per_MWh', 'Fuel_Price', 'Wind_MW', 'Solar_MW',
                      'Demand_DLAP_MW', 'Demand_MW']]


# Concantonate weather dataframes
weather_df = pd.concat([NREL_df, NOAA_obs_df, NOAA_forc_df], axis=0)

# Join onto date_range to show missing datetimes
final_df = date_range.merge(weather_df, how='left', left_index=True, right_index=True)
final_df= final_df.merge(CAISO_df, how='left', left_index=True, right_index=True)


# Create columns for date components 
final_df["Year"] = final_df.index.year
final_df["Month"] = final_df.index.month 
final_df["Day"] = final_df.index.day
final_df["Hour"] = final_df.index.hour 
final_df["Weekday"] = final_df.index.to_series().apply(lambda x:x.weekday).astype(int)
final_df["Weekend"] = ((pd.DatetimeIndex(final_df.index).dayofweek) // 5 == 1).astype(int)
final_df["Time"] = final_df.index.to_series().apply(lambda x:x.time)
final_df['INTERVALSTARTTIME_PST'] = final_df.index.to_series().apply(lambda x: x-datetime.timedelta(hours=7, minutes=00))
final_df['INTERVALSTARTTIME_PST'] = final_df['INTERVALSTARTTIME_PST'].apply(daylight_savings_correction)


print('Time to finish: ', (time.time()-start_time)/60)

final_df.to_csv('final_df.csv')




