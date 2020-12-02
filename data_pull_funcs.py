# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:56:10 2020

@author: bnebe
"""

############# DO NOT DELETE ANY IMPORTS ############
# import relevant functionality from caiso_data_manager

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
import math
import random
import pgeocode
import eeweather
import Credentials
import time
from datetime import date
from datetime import timedelta
from functools import reduce 
from caiso_data_manager import CAISO 
from caiso_data_manager import daylight_savings_correction
from caiso_data_manager import daterange 
from caiso_data_manager import end_day_of_month
from noaa_sdk import noaa
from itertools import tee
from datetime import datetime
import shutil


def caiso_data_pull(output_folder, start_year, start_month, start_day, end_year, end_month, end_day, node, area_name, trading_hub):
    """
    Collect CAISO data frome start to end date

    :param string output_folder:
    :param integer start_year:
    :param integer start_month:
    :param integer start_day:
    :param integer end_year:
    :param integer end_month:
    :param integer end_day:
    :param string node:
    :param string area_name:
    :param list/string trading_hub:
    :return: DataFrame of CAISO data from start to end dates
    :rtype: DataFrame
    """
    #============================================================================================================================================================================#
        
    # API version - Don't change 
    version = 1
    result_format = 6 
    
    #============================================================================================================================================================================#
    print('Removing Old Data')
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    
    #============================================================================================================================================================================#
    print('Date Handling')
    # Create an instance of the CAISO class 
    caiso = CAISO(
        output_folder=output_folder,
        start_date=date(start_year, start_month, start_day),
        end_date=date(end_year, end_month, end_day),
        datasets=["LMP", "FUEL_PRICES", "CAISO_DEMAND_FORECAST", "WIND_AND_SOLAR_FORECAST"]
    ) 
    
    # CAISO limits the data pull to a month at a time, so prepare a list of unique months 
    month_begin_days = pd.date_range(start=caiso.start_date, end=caiso.end_date, freq="MS")
    month_end_days = pd.date_range(start=caiso.start_date, end=caiso.end_date, freq="M")
    
    # Get current directory 
    current_directory = os.getcwd()
    
    #============================================================================================================================================================================#
    
    print('LMP Price Data')
    # User Inputs about LMP Price 
    query_name="PRC_LMP"
    market_run_id="DAM"
    
    caiso.set_lmp_price(
        query_name=query_name,
        market_run_id=market_run_id,
        node=node, 
        begin_days=month_begin_days, 
        end_days=month_end_days
    )
    
    #============================================================================================================================================================================#
    
    print('Demand Data Fetching')
    # User Inputs about Demand data 
    query_name="SLD_FCST"
    market_run_id="DAM"
    
    # Get Demand Data 
    caiso.set_demand(
        query_name=query_name, 
        market_run_id=market_run_id,
        begin_days=month_begin_days, 
        end_days=month_end_days
    )
    
    #============================================================================================================================================================================#
    
    print('Fuel Price Fetching')
    # User Inputs about Fuel Price data 
    query_name="PRC_FUEL"
    fuel_region_id="FRPGE1"
    
    # Get Fuel Price Data 
    caiso.set_fuel_prices(
        query_name=query_name, 
        fuel_region_id=fuel_region_id, 
        begin_days=month_begin_days, 
        end_days=month_end_days
    )
    
    #============================================================================================================================================================================#
    
    print('Renewables Fetching')
    # User Inputs about Renewables 
    query_name = "SLD_REN_FCST"
    market_run_id = "DAM"
    
    # Get wind_solar_forecast 
    caiso.set_wind_solar_forecast(
        query_name=query_name, 
        market_run_id=market_run_id, 
        begin_days=month_begin_days, 
        end_days=month_end_days
    )
    
    #============================================================================================================================================================================#
    
    print('Filter Columns, Prepare Data')
    # Define the desired columns to filter the datasets by 
    required_columns = ["INTERVALSTARTTIME_GMT","INTERVALENDTIME_GMT", "OPR_DT", "OPR_HR", "MW"]
    required_columns_fuel = ['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT', 'OPR_DT', 'OPR_HR', 'PRC']
    
    lmp_price_clean_df = caiso.pre_process_lmp_prices(df=caiso.raw_lmp_price_df, desired_columns=required_columns)
    fuel_price_clean_df = caiso.pre_process_fuel_prices(df=caiso.raw_fuel_price_df, desired_columns=required_columns_fuel)
    demand_clean_df = caiso.pre_process_demand(df=caiso.raw_demand_df, desired_columns=required_columns)
    demand_clean_node = caiso.pre_process_demand(df=caiso.raw_demand_df, area_name=area_name, desired_columns=["INTERVALSTARTTIME_GMT","INTERVALENDTIME_GMT", "OPR_DT", "OPR_HR", "MW", "TAC_AREA_NAME"]) #Changed desired columns to include Area
    wind_clean_df = caiso.pre_process_renewable_data(
        df=caiso.raw_wind_solar_forecast_df, 
        desired_columns=["INTERVALSTARTTIME_GMT","INTERVALENDTIME_GMT", "OPR_DT", "OPR_HR", "MW", "TRADING_HUB"], #Changed desired columns to include Trading Hub
        trading_hub=trading_hub,
        renewable_type=["Wind"]
    )
    solar_clean_df = caiso.pre_process_renewable_data(
        df=caiso.raw_wind_solar_forecast_df, 
        desired_columns=["INTERVALSTARTTIME_GMT","INTERVALENDTIME_GMT", "OPR_DT", "OPR_HR", "MW", "TRADING_HUB"], #Changed desired columns to include Trading Hub
        trading_hub=trading_hub, 
        renewable_type=["Solar"]
    )
    
    #============================================================================================================================================================================#
    
    print('Prepare Unified Dataset')
    # Define start and end columns for joining 
    start = "INTERVALSTARTTIME_GMT"
    end = "INTERVALENDTIME_GMT" 
    
    # Sum wind and solar trading zones by time
    wind_clean_df['Wind_MW'] = wind_clean_df['MW'].groupby(wind_clean_df['INTERVALSTARTTIME_GMT']).transform('sum') #NEW Sums MW for all trading hubs per time stamp
    solar_clean_df['Solar_MW'] = solar_clean_df['MW'].groupby(solar_clean_df['INTERVALSTARTTIME_GMT']).transform('sum') #NEW Sums MW for all trading hubs per time stamp
        
    # Subset columns that are used in the model and common 
    lmp_prices_final_df = lmp_price_clean_df[['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT', 'OPR_DT', 'OPR_HR', 'MW']]
    lmp_prices_final_df.columns = ['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT', 'OPR_DT', 'OPR_HR', 'LMP_Price_Per_MWh']
    fuel_final_df = fuel_price_clean_df[['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT', 'PRC']] #NEW Added Fuel for fuel price
    fuel_final_df.columns = ['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT', 'Fuel_Price'] #NEW Added Fuel for fuel price
    demand_final_df = demand_clean_df[["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "MW"]]
    demand_final_df.columns = ["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "Demand_MW"]
    demand_final_node = demand_clean_node[['INTERVALSTARTTIME_GMT', "INTERVALENDTIME_GMT", 'MW', "TAC_AREA_NAME"]] #NEW Added for area demand
    demand_final_node.columns = ['INTERVALSTARTTIME_GMT', "INTERVALENDTIME_GMT", 'Demand_DLAP_MW', 'Area_Name'] #NEW Added for area demand
    wind_final_df = wind_clean_df[["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "Wind_MW"]]
    wind_final_df.columns = ["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "Wind_MW"]
    solar_final_df = solar_clean_df[["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "Solar_MW"]]
    solar_final_df.columns = ["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "Solar_MW"]
    
    wind_final_df = wind_final_df.drop_duplicates() #NEW Removes duplicate time stamps
    solar_final_df= solar_final_df.drop_duplicates() #NEW Removes duplicate time stamps
    
    
    # Merge all the datasets 
    list_dataframes_to_merge = [wind_final_df, solar_final_df, demand_final_df, fuel_final_df, lmp_prices_final_df, demand_final_node]
    input_df = reduce(lambda left,right: pd.merge(left, right, on=[start, end], how="inner"), list_dataframes_to_merge)
    
    # Clean up the order of the resultant dataframe 
    input_df = input_df[[
        'INTERVALSTARTTIME_GMT', 
        'INTERVALENDTIME_GMT', 
        'OPR_DT', 
        'OPR_HR', 
        'LMP_Price_Per_MWh',
        'Fuel_Price', #NEW (from fuel final)
        'Wind_MW',
        'Solar_MW',
        'Demand_DLAP_MW', #NEW (from demand final node)
        'Demand_MW'
    ]]
    
    #============================================================================================================================================================================#
    
    print('Index GMT/UTC Time')
    # Convert time zone to PST and apply day light savings correction 
    input_df["INTERVALSTARTTIME_GMT"] = pd.to_datetime(input_df["INTERVALSTARTTIME_GMT"])
    input_df['INTERVALSTARTTIME_PST'] = input_df['INTERVALSTARTTIME_GMT'].apply(lambda x: x-timedelta(hours=7, minutes=00))
    input_df['INTERVALSTARTTIME_PST'] = input_df['INTERVALSTARTTIME_PST'].apply(daylight_savings_correction)
        
    # Set Index using the timestamps 
    input_df.set_index("INTERVALSTARTTIME_GMT", inplace=True)
    return input_df


def create_dict_solar_zero_hours(solar_zero_df, thresh):
    solar_zero_df['hour'] = solar_zero_df.index.hour
    solar_zero_df['month'] = solar_zero_df.index.month
    solar_zero_df = solar_zero_df.groupby(by=['month', 'hour']).sum()
    solar_zero_df = solar_zero_df[solar_zero_df.iloc[:, 0] < thresh]
    solar_zero_df = solar_zero_df.reset_index()
    
    hour_dict = {}
    for i in range(12):
        hour_dict[i+1] = solar_zero_df[solar_zero_df['month'] == i+1]['hour'].values.tolist()

    return hour_dict


#Pythagorean distance between two points passed in as arrays
def calculateDistance_arr(arr1, arr2):
    """
    Returns Pythagorean distance between two points

    :param array arr1: first point as (2,1) array
    :param array arr2: second points as (2,1) array
    :return: distance between two coordinates 
    :rtype: float
    """
    x1, y1 = arr1[0], arr1[1]
    x2, y2 = arr2[0], arr2[1]
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist


def iterative_farthest_sample(points, k):
    """
    Returns sample set of data using greedy iterative furthest point sampling

    :param array points: array of points to sample from as (n,2) array (x,y)
    :param integer k: number of points to sample from array
    :return: array of sample datapoints 
    :rtype: array
    """
    remaining_points = points[:]
    solution_set = np.empty(shape=(0,2))
    solution_set = np.append(solution_set, remaining_points[(random.randint(0, len(remaining_points) - 1))].reshape(1,2), axis=0)
    
    for _ in range(k-1):
        distances = [calculateDistance_arr(p, solution_set[0]) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], calculateDistance_arr(p, s))
        solution_set = np.append(solution_set, 
                                 remaining_points[distances.index(max(distances))].reshape(1,2), axis=0)
    return solution_set


def check_noaa_standard(arr, length_thresh, range_thresh, interval_thresh):
    """
    Returns TRUE/FALSE if noaa.get_observations_by_lat_lon meets criteria given

    :param array arr: array of points to judge criteria on
    :param float length_thresh: threshold for generators length
    :param float range_thresh: threshold for generators date range
    :param float interval_thresh: threshold for generators interval (length/range)
    :return: bool if points meet criteria
    :rtype: boolean
    """
    n = noaa.NOAA()
    gen, gen_temp = tee(n.get_observations_by_lat_lon(arr[0], arr[1]))
    gen_length = len(list(gen_temp))
    print('Length:', gen_length)
    if gen_length < length_thresh:
        return False
    
    temp_lst = []
    for i in gen:
        temp_lst.append(i['timestamp'])
    
    date_lst = [datetime.strptime(x,'%Y-%m-%dT%H:%M:%S%z') for x in temp_lst]
    date_length = len(date_lst)
    
    date_range = (max(date_lst)-min(date_lst)).total_seconds()/3600
    date_interval = date_length/date_range
    
    
    print('Range:', date_range)
    print('Interval:', date_interval)
    if date_range > range_thresh and date_interval >= interval_thresh:
        return True
    else:
        return False


def iterative_farthest_sample_check(points, k, check=False, length_thresh=336, range_thresh=336, interval_thresh=1):
    """
    Returns sample set of data using greedy iterative furthest point sampling

    :param array points: array of points to sample from as (n,2) array (x,y)
    :param integer k: number of points to sample from array
    :return: array of sample datapoints 
    :rtype: array
    """
    remaining_points = points[:]
    solution_set = np.empty(shape=(0,2))
    
    if check == True:
        while True:
            rand_int = (random.randint(0, len(remaining_points) - 1))
            temp_point = remaining_points[rand_int]
            if check_noaa_standard(temp_point, length_thresh, range_thresh, interval_thresh) == True:
                print('Selected initial point')
                break
            else:
                print('Deleteing initial point')
                remaining_points = np.delete(remaining_points, rand_int, axis=0)
    
        solution_set = np.append(solution_set, temp_point.reshape(1,2), axis=0)
        
    else:
        solution_set = np.append(solution_set, remaining_points[(random.randint(0, len(remaining_points) - 1))].reshape(1,2), axis=0)
    
    for _ in range(k-1):
        distances = [calculateDistance_arr(p, solution_set[0]) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], calculateDistance_arr(p, s))
        if check == True:
            while True:
                print()
                temp_point = remaining_points[distances.index(max(distances))]
                if check_noaa_standard(temp_point, length_thresh, range_thresh, interval_thresh) == True:
                    print('Selected next point')
                    break
                else:
                    print('Delete next point and distance')
                    remaining_points = np.delete(remaining_points, distances.index(max(distances)), axis=0)
                    distances.remove(max(distances))
                    
        solution_set = np.append(solution_set, remaining_points[distances.index(max(distances))].reshape(1,2), axis=0)
    return solution_set


#Sums distances betweeen all points to evaluate model
def evaluate_solution(solution_set):
    """
    Evaluates array based on total sum distance of all points

    :param array solution_set: array of points to evaluate as (n,2) array (x,y)
    :return: Total sum of distance between all points
    :rtype: float
    """
    return sum([calculateDistance_arr(a, b) for a, b in zip(solution_set[:-1], solution_set[1:])])

#Run iterative_farthest_sample over a certain number of tries, selects the sample with the lowest summed distance
def optimal_furthest_sample(points, k, tries):
    """
    Returns sample set of data using greedy iterative furthest point sampling
    Chooses best sample based on inputed tries

    :param array points: array of points to sample from as (n,2) array (x,y)
    :param integer k: number of points to sample from array
    :param integer tries: number runs to find optimal sample set
    :return: array of sample datapoints, optomized on total sum distance
    :rtype: array
    """
    solution_sets = [iterative_farthest_sample(points, k) for _ in range(tries)]
    sorted_solutions = sorted(solution_sets, key=evaluate_solution, reverse=False)
    return sorted_solutions[0]


def convert_zips_to_coor(zips):
    """
    Converts zip codes to coordinates, latitude-longitude

    :param dataframe zips: dataframe of zipcodes to convert to coordinates
    :return: array of coordinates based on zipcodes
    :rtype: array
    """
    nomi = pgeocode.Nominatim('US')
    zips_to_lat = []
    zips_to_lon = []
    
    for i in zips.index:
        x = nomi.query_postal_code(str(zips.loc[i][0]))
        zips_to_lat.append(x['latitude'])
        zips_to_lon.append(x['longitude'])
    
    points = np.column_stack((zips_to_lat, zips_to_lon))
    return points


def convert_coor_to_stations(sample, n_stations):
    """
    Converts coordinates to nearest, high quality weather stations, removing duplicates

    :param array sample: sample of coordinates to be converted to nearest station
    :param integer n_stations: number of stations to grab per sample point
    :return: array of coordinates for n_stations nearest weather stations
    :rtype: array
    """
    station_lat = []
    station_lon = []
    
    for i in sample:
        x = eeweather.rank_stations(i[0], i[1])
        for n in range(n_stations):
            station_lat.append(x[x['rough_quality'] == 'high']['latitude'][n])
            station_lon.append(x[x['rough_quality'] == 'high']['longitude'][n])
    
    station_coor = np.column_stack((station_lat, station_lon))
    station_coor = np.unique(station_coor, axis=0)
    return station_coor


def is_leap_year(year):
    """
    Checks if given year is leap year, returns tring to be used in NREL lookup url

    :param integer year: year to check if leap year
    :return: string TRUE/FALSE to be used in NREL lookup
    :rtype: string
    """
    int_year = int(year)
    if (int_year % 4) == 0:
        if (int_year % 100) == 0:
            if (int_year % 400) ==0:
                leap_year = 'true'
            else:
                leap_year = 'false'
        else:
            leap_year = 'true'
    else:
        leap_year = 'false'
    return leap_year


def NREL_station_lookup(lat, lon, year, interval='60', utc='false'):
    """
    Returns dataframe of NREL weather observations for the given latitude, longitude, and year

    :param float lat: lat to lookup weather data on
    :param float lon: lon to lookup weather data on
    :param integer year: year to look date up in
    :param string interval: 30 or 60 intervals for weather observations (string to fit in url)
    :param string utc: TRUE/FALSE to convert observations to UTC (string to fit in url)
    :return: dataframe of NREL weather observations for that year
    :rtype: dataframe
    """
    cred = Credentials.Credentials()
    
    leap_year = is_leap_year(year)
        
    # According to site pressure could also be added but was unable to work it out
    # https://nsrdb.nrel.gov/data-sets/api-instructions.html
    attributes = 'wind_speed,air_temperature,solar_zenith_angle,wind_direction,dew_point,relative_humidity'
    attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle,wind_direction,dew_point,relative_humidity'
    
    api_key = cred.api_key()
    your_name = cred.your_name()
    reason_for_use = cred.reason_for_use()
    your_affiliation = cred.your_affiliation()
    your_email = cred.your_email()
    mailing_list = cred.mailing_list()

    url = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)

    dftemp = pd.read_csv(url, skiprows=2)
    dftemp['Wind Speed'] = dftemp['Wind Speed'] * 2.236936 # Convert wind speed from m/s to mph
    dftemp['lat'] = lat
    dftemp['lon'] = lon
    # Set the time index in the pandas dataframe:
    dftemp = dftemp.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=525600/int(interval)))
    return dftemp


def NREL_combine_obs(sample, year):
    """
    Average NREL weather observations for given station coordinates and year

    :param array sample: coordinates for weather stations used for averaging 
    :param integer year: year to use for lookup
    :return: dataframe of averaged weather for stations and year
    :rtype: dataframe
    """    
    sample_weather_df = pd.DataFrame()
    for row in sample:
        tempdf = NREL_station_lookup(row[0], row[1], year=year)
        sample_weather_df = sample_weather_df.append(tempdf)

    #Average weather across all stations
    avg_weather_df = sample_weather_df.groupby(by=sample_weather_df.index).mean()
    avg_weather_df = avg_weather_df.tz_localize('UTC', level=0, nonexistent='NaT', ambiguous='NaT')
    return avg_weather_df


# NOAA.get_observations_by_lat_lon uses GMT time 10/09/2020
def NOAA_combine_obs(sample, replace_list, remove_list):
    """
    Average NOAA weather observations for given station coordinates and year
    
    :param array station_coor: coordinates for weather stations used for averaging 
    :param integer year: year to use for lookup
    :return: dataframe of averaged weather observations for stations and year
    :rtype: dataframe
    """    
    n = noaa.NOAA()
    obs_df = pd.DataFrame()    
    
    for row in sample:
        observations = n.get_observations_by_lat_lon(row[0], row[1])
        for i in observations:
            for j in remove_list:
                del i[j]
            for j in replace_list:
                i[j] = i[j]['value']
            i['lat'] = row[0]
            i['lon'] = row[1]
            
            obs_df = obs_df.append(i, ignore_index=True)
    
    obs_df['timestamp'] = pd.to_datetime(obs_df['timestamp'])
    obs_df['timestamp_rounded'] = obs_df['timestamp'].dt.round(freq = 'H')
    # obs_df['timestamp_rounded'] = obs_df['timestamp_rounded'].dt.tz_localize('utc')
    obs_df.set_index(keys=['lat', 'lon', 'timestamp_rounded'], drop=True, inplace=True)
    
    
    obs_df = obs_df[~obs_df.index.duplicated(keep='first')]
    obs_df.drop(columns = ['timestamp'], inplace=True)
    obs_df.reset_index(level=['lat', 'lon'], drop=True, inplace=True)
    obs_df[obs_df.columns] = obs_df[obs_df.columns].astype(float)
    
    NOAA_obs_df = obs_df.groupby(by=obs_df.index).mean()
    
    return NOAA_obs_df



def NOAA_combine_forc(sample, keep_list):
    """
    Average NOAA weather forecasts for given station coordinates and year
    NOAA.points_forecast uses local (PST) time 10/09/2020

    :param array sample: coordinates for weather stations used for averaging 
    :param list keep_list: list of headers wished to retain
    :return: dataframe of averaged weather forecasts for stations and year
    :rtype: dataframe
    """
    n = noaa.NOAA()    
    forc_df = pd.DataFrame()
    
    for row in sample:
        gen = n.points_forecast(row[0], row[1])
        for i in gen['properties']['periods']:
            i['lat'] = row[0]
            i['lon'] = row[1]
            forc_df = forc_df.append(i, ignore_index=True)
    
    forc_df = forc_df[keep_list]
    
    # Convert time to UTC
    forc_df['startTime'] = pd.to_datetime(forc_df['startTime'])
    forc_df['startTime'] = forc_df['startTime'].dt.tz_convert('utc')
    forc_df.set_index(keys=['startTime'], drop=True, inplace=True)
    
    # Find average of wind speeds
    temp_df = forc_df['windSpeed'].str.split(' to ', expand=True)
    
    temp_df[0] = temp_df[0].str.slice(start=0, stop=2)
    temp_df[1] = temp_df[1].str.slice(start=0, stop=2)
    temp_df[temp_df.columns] = temp_df[temp_df.columns].astype(float) 
    temp_df['avg'] = temp_df.mean(axis=1)
    forc_df['windSpeed'] = temp_df['avg']
    
    # Count instances of index
    temp_df['ct'] = 1
    temp_df2 = temp_df.groupby(by=temp_df.index).sum()
    temp_df2 = temp_df2['ct']
    forc_df = forc_df.merge(temp_df2, how='left', left_index=True, right_index=True)
    forc_df = forc_df[forc_df['ct']>9]
    
    
    NOAA_forc_df = forc_df.groupby(by=forc_df.index).mean()
    NOAA_forc_df['temperature'] = (NOAA_forc_df['temperature'] - 32)*(5/9) # Convert F to C
    
    return NOAA_forc_df