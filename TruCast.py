import os
import argparse
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



EXCEL_FILE_PATH = 'CBO Revenue Short.xlsx'
OUTPUT_FILE_PATH = 'TruCast Output 3 Month.xlsx'
MEDIUM_THRESHOLD = 200000 # in dollars
LARGE_THRESHOLD = 533000 # in dollars
NUMBER_OF_MONTHS = 12 # in months

def fixed(revenue_series):
    return [revenue_series.iloc[-1]]*NUMBER_OF_MONTHS

def three_month(revenue_series):
    return [(revenue_series.iloc[-3] + revenue_series.iloc[-2] + revenue_series.iloc[-1])/3]*NUMBER_OF_MONTHS


def ARIMA_rolling_forcast_origin(revenue_series, number_of_predicted_months, p, q, d):
    # Preforms a rolling forcast origin for an using an arima model on times series data for a set number of number_of_predicted_months < len(revenue_series) 
    arima_revenue_projection_list = revenue_series.tolist()
    total_revenue_by_month_list = revenue_series.tolist()
    for index, revenue in enumerate(arima_revenue_projection_list):
        if len(arima_revenue_projection_list) - index < number_of_predicted_months:
            try:
                model = ARIMA(total_revenue_by_month_list[0:index], order=(p, d, q))
                try:
                    results = model.fit()
                    forecast = results.forecast(steps=1)
                except LinAlgError as e:
                    forecast = [-10000000]
                arima_revenue_projection_list[index] = forecast[0]
            except LinAlgError as e:
                arima_revenue_projection_list[index] = np.nan
                continue
    arima_revenue_projection = pd.Series(data=arima_revenue_projection_list, index=revenue_series.index)
    
    return arima_revenue_projection

def arima(revenue_series):
    # Train on 1/3 of the data
    number_of_predicted_months = int(len(revenue_series)/3)

    # Choose the range of p and q that you want to optmize over
    p_range = 6
    q_range = 6
    d = 1

    # Create an ARIMA model on the data for each value of p and q forcast it forward using a rolling origin forcast, determine which pair of p and q works
    # best and output that
    arima_revenue_projection_list = revenue_series.tolist()
    projection_percent_difference = {}
    for i in range(1, p_range):
        for ii in range(1, q_range):
            projection = ARIMA_rolling_forcast_origin(revenue_series, number_of_predicted_months, i, ii, d)

            percent_difference = abs(projection[-number_of_predicted_months:] - arima_revenue_projection_list[-number_of_predicted_months:]) / arima_revenue_projection_list[-number_of_predicted_months:] * 100
            projection_avg_percent_difference = percent_difference.mean()

            if projection_avg_percent_difference == np.nan:
                # If any errors occur set the percent error to really high so it doesn't get choosen
                projection_percent_difference[(i, ii)] = 1000
            else:
                projection_percent_difference[(i, ii)] = projection_avg_percent_difference

    min_error_key = min(projection_percent_difference, key=projection_percent_difference.get)
    p = min_error_key[0]
    q = min_error_key[1]

    # Use the ARIMA model that produced the minimum error and forcast 1 time step forward
    model = ARIMA(revenue_series, order=(p, d, q))
    results = model.fit()
    forecast = results.forecast(steps=NUMBER_OF_MONTHS)

    return forecast

def prophet(revenue_series):
    # for the model, make a dataframe with columns 'ds' and 'y' out of the hospital row, which was a Series
    curr_hospital = pd.DataFrame({'ds':revenue_series.index, 'y':revenue_series.values})

    # make the months datetime objects, and the revenues numbers
    curr_hospital['ds'] = pd.to_datetime(curr_hospital['ds'])
    curr_hospital['y'] = pd.to_numeric(curr_hospital['y'])

    # training on all revenues up to the last year (CAN BE MODIFIED TO BE THE LAST MONTH, TWO MONTHS, ETC.)
    # MUDIT COMMENT: JUST GOT RID OF THE "-NUMBER_OF_MONTHS" HERE, NOW TRAINING ON ALL DATA IN REVENUE_SERIES
    train = curr_hospital.iloc[:len(curr_hospital)]

    # testing how accurately we predict the last year (CAN BE MODIFIED TO BE THE LAST MONTH, TWO MONTHS, ETC.)
    test = curr_hospital.iloc[len(curr_hospital) - NUMBER_OF_MONTHS:]

    # if this row of the dataframe contains less than two non-nan values, we cannot predict on it — SKIP
    if train[train['y'].notnull()].shape[0] < 2:
        return None # WE SHOULD REDIRECT HERE TO ANOTHER MODEL, IF THERE IS NOT ENOUGH TRAINING DATA FOR PROPHET
    
    # fit the model, generate the forecast
    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods = NUMBER_OF_MONTHS, freq='MS')
    forecast = m.predict(future)

    # generate predictions (contains all from 2004 through 2023)
    predictions = forecast[['ds', 'yhat']].set_index('ds')['yhat'].iloc[-NUMBER_OF_MONTHS:]
    predictions_lower = forecast[['ds', 'yhat_lower']].set_index('ds')['yhat_lower'].iloc[-NUMBER_OF_MONTHS:]
    predictions_upper = forecast[['ds', 'yhat_upper']].set_index('ds')['yhat_upper'].iloc[-NUMBER_OF_MONTHS:]
    
    output_columns = predictions.index

    # return a tuple of the lower, point estimate, and upper bound for the last month
    return (predictions.values, predictions_lower.values, predictions_upper.values)

def determine_projection(revenue_series, medium_threshold, large_threshold, fixed_rate_projection, three_month_projection, arima_projection, prophet_projection):
    # if it's fixed, return fixed rate
    if (fixed_rate_projection != -1):
        return fixed_rate_projection
    
    if revenue_series.iloc[-NUMBER_OF_MONTHS:].count() == NUMBER_OF_MONTHS:
        yearly_revenue = sum(revenue_series.iloc[-NUMBER_OF_MONTHS:])
    else:
        yearly_revenue = (revenue_series.mean())*NUMBER_OF_MONTHS
    
    if yearly_revenue < medium_threshold:
        return arima_projection
    else:
        return prophet_projection
    
def determine_projection_type(revenue_series, medium_threshold, large_threshold):
    if (revenue_series.iloc[-2] == revenue_series.iloc[-1]) or (np.isnan(revenue_series.iloc[-1])):
        return 'fixed'
    elif revenue_series.iloc[-NUMBER_OF_MONTHS:].count() == NUMBER_OF_MONTHS:
        yearly_revenue = sum(revenue_series.iloc[-NUMBER_OF_MONTHS:])
    else:
        # if there isn't a years worth of data just return a 3 month average
        return 'three_month'
    
    if yearly_revenue < medium_threshold:
        return 'arima'
    else:
        return 'prophet'
    
def rename_duplicates(names):
    name_count = {}
    modified_names = []

    for name in names:
        if name in name_count:
            name_count[name] += 1
        else:
            name_count[name] = 1

        if name_count[name] == 1:
            modified_names.append(name)
        else:
            modified_names.append(f"{name}_{name_count[name]-1}")

    return modified_names
    
def process(input_path, export_path):
    if input_path.endswith('.xls') or input_path.endswith('.xlsx'):
        #revenue_data = pd.read_excel(EXCEL_FILE_PATH, index_col=['Site', 'Customer Code'])
        revenue_data = pd.read_excel(input_path, index_col='site')
    else:
        print("Error: Input file must be in Excel format.")

    # TODO Clean data


    # Create dataframe for the output date
    new_months = list(pd.date_range(revenue_data.columns[-1], periods=13, freq='M').strftime('%Y-%m'))[1:13]
    revenue_data.columns = [col.strftime('%Y-%m') if isinstance(col, pd.Timestamp) else col for col in revenue_data.columns]
    column_names = list(revenue_data.columns) + new_months
    revenue_data.index = rename_duplicates(revenue_data.index)
    output_data = pd.DataFrame(index=revenue_data.index, columns=column_names)

    #Loop through hospitals
    for hospital, revenue_series in revenue_data.iterrows():
        projection_type = determine_projection_type(revenue_series, MEDIUM_THRESHOLD, LARGE_THRESHOLD)
        print(f'{hospital}: {projection_type}')
        if projection_type == 'fixed':
            projection = fixed(revenue_series)
        elif projection_type == 'three_month':
            projection = three_month(revenue_series)
        elif projection_type == 'arima':
            try:
                projection = arima(revenue_series)
            except:
                projection = three_month(revenue_series)
            #projection = [0]*12
        elif projection_type == 'prophet':
            try:
                (projection, prophet_low, prophet_high) = prophet(revenue_series)
            except:
                projection = three_month(revenue_series)

        projection = [rev if rev >= 0 or np.isnan(rev) else 0 for rev in projection] # Sets a lower bound of 0 for any projection

        output_data.loc[hospital] = pd.concat([revenue_series, pd.Series(data=projection, index=new_months)])

    output_data.dropna(inplace=True, how='all')
    output_data.to_excel(export_path, index_label='site')
    return output_data

