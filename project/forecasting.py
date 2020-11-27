import os
import random
import math
from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

def plot_ts(results, country):
    print(results)
    fig = go.Figure(go.Scatter(
        x = results.index.values, y = results,
        mode = 'lines'
    ))
    fig.update_layout(title_text = 'Future Air Quality of ' + country.upper(),
                    xaxis = dict(title = 'Forecast 2 years into the future', showticklabels = False),
                    yaxis = dict(title = 'Air Quality Concentration'))
    fig.write_image('results/Forecast_'+country+'.png')

def clean(ts):
    ts['Time'] = pd.to_datetime(ts['utc'], format = '%Y-%m-%d')
    ts = ts.drop(['utc'], axis = 1)
    ts.set_index('Time', inplace = True)
    ts['value'] = ts['value'].astype('float32')
    median = np.median(ts[ts.value != -999])
    ts = ts.replace(-999, median)
    return ts

def fit_ts(ts,train, val):
    train = train.sample(frac=0.5, weights = [1/len(train) for i in range(0,len(train))]).sort_index(ascending = True)
    val = val.sample(frac=0.5, weights = [1/len(val) for i in range(0,len(val))]).sort_index(ascending = True)
    windows = [3,4,5]
    orders = [(1,0,0),(1,1,1),(1,2,1),(1,1,2),(1,2,2),(2,1,1)]
    max_score, best_pair = np.Inf, None
    for window in windows:
        for order in orders:
            train['RollingAvg'] = train.loc[:,'value'].rolling(window = window).mean()
            val['RollingAvg'] = val.loc[:,'value'].rolling(window = window).mean()
            train,val = train.dropna(), val.dropna()
            model = SARIMAX(endog = train['value'], order = order)
            results = model.fit()
            preds = results.forecast(steps = len(val))
            rmse = math.sqrt(mean_squared_error(val['value'],preds))
            if (rmse < max_score):
                max_score = rmse
                best_pair = [window,order]
    ts = ts.sample(frac=0.5, weights = [1/len(ts) for i in range(0,len(ts))]).sort_index(ascending = True)
    ts['RollingAvg'] = ts.loc[:,'value'].rolling(window = best_pair[0]).mean()
    ts = ts.dropna()
    tuned_model = SARIMAX(endog = ts['value'], order = best_pair[1])
    tuned_model = tuned_model.fit(maxlags = 15, ic = 'aic')
    return tuned_model

def train_test(ts):
    split = math.floor(len(ts)*0.7)
    train, val = ts.iloc[0:split,:], ts.iloc[split:len(ts),:]
    predictor = fit_ts(ts,train, val)
    future = predictor.predict(0,731)[1:]
    return future

def main():
    countries = ['bd','bh','ca','gt','hk','hu','id','in','iq','kw','lk','lu','mn','no',
                'np','pe','pk','ug','vm']
    predictions = []
    for country in countries:
        ts = pd.read_csv('data/air_quality_updated/'+country+'_data.csv', header = 0, index_col = None).filter(['utc','value'],axis=1)
        ts = clean(ts)
        results = train_test(ts)
        predictions.append(results)
        plot_ts(results, country)

print(os.getcwd())
main()
