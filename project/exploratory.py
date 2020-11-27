import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopy.distance
from scipy.stats import pearsonr

colours = [255*i/19 for i in range(0,19)]

def get_missing(df, feat):
    df = df[feat]
    print('For the feature:',feat,', this is the number of missing values.')
    num_missing = df[df == '..'].count()
    print(num_missing,'\n')
    return num_missing

def get_stats(df, feat):
    missing = get_missing(df,feat)
    df = df[feat]
    df = df[df != '..'].astype('float32')
    stats = [feat,np.mean(df),np.median(df),min(df),max(df),np.var(df),missing]
    return stats

def heatmap(df):
    '''Add in title'''
    correlates = []
    for featA in df.columns[2:]:
        a = df[featA].to_numpy('float32')
        curr_corr = []
        for featB in df.columns[2:]:
            v = df[featB].to_numpy('float32')
            curr_corr.append(pearsonr(a,v)[0])
        correlates.append(curr_corr)
    df = df.drop(['Country','Year'], axis = 1)
    fig = go.Figure(data = go.Heatmap(
        z = correlates, x = df.columns, y = df.columns
    ))
    fig.update_layout(title_text = 'Correlation Heat Map of Features')
    fig.write_image('figures/Heatmap.png')

def histograms(df, feats):
    '''Add in axis labels and title for each figure'''
    for feat in feats:
        current = df[feat]
        fig = go.Figure(data = go.Histogram(
            x = current
        ))
        fig.update_layout(title_text = 'Histogram of ' + feat,
                        xaxis_title_text = feat,
                        yaxis_title_text = 'Count')
        fig.write_image('figures/Histogram_'+feat+'.png')

def timeseries(df, feats):
    '''Add in plot formatting.'''
    years = [i for i in range(2000,2019)]
    for feat in feats:
        fig = go.Figure()
        current = df[['Country',feat]]
        annotations = []
        for country,i in zip(current.Country.unique(),range(0,19)):
            fig.add_trace(go.Scatter(
                x = years, y = current[feat][current.Country == country],
                mode = 'lines',
                name = country
            ))
            # annotations.append(dict(xref='paper',x=0.95,
            #                         y=current[feat][current.Country == country].iloc[-1],
            #                         xanchor='left',yanchor='middle',
            #                         text=country
            # ))
        fig.update_layout(title_text = 'Time Series plot of ' + feat,
                        # annotations = annotations,
                        showlegend = True)
        fig.write_image('figures/TimeSeries_'+feat+'.png', width = 1050, height = 750)

def distancemap(df):
    distances = []
    for countryA in df.Country.unique():
        curr_country = []
        coordsA = tuple(df[['Latitude','Longitude']][df.Country == countryA].to_numpy()[0])
        for countryB in df.Country.unique():
            coordsB = tuple(df[['Latitude','Longitude']][df.Country == countryB].to_numpy()[0])
            curr_country.append(geopy.distance.distance(coordsA,coordsB).km)
        distances.append(curr_country)

    fig = go.Figure(data = go.Heatmap(
        z = distances, x = df.Country.unique(), y = df.Country.unique()
    ))
    fig.update_layout(title_text = 'Heat Map of the Inter-Country Distances')
    fig.write_image('figures/Distance_Heatmap.png')

def movingavg(df,feats):
    window = 2
    fig = make_subplots(rows = 5, cols = 5)
    pairs = [[i,j] for i in range(1,6) for j in range(1,6)]
    for feat,pair in zip(feats,pairs):
        current = df[['Year',feat]]
        averages = []
        for year in df.Year.unique():
            averages.append(np.mean(current[feat][current.Year == year].to_numpy('float32')))
        maverage = []
        for i in range(2,19):
            maverage.append((averages[i] + averages[i-1] + averages[i-2])/3)

        fig.append_trace(go.Scatter(
            x = [i for i in range(1,len(maverage)+1)], y = maverage, name = feat
        ), row = pair[0], col = pair[1])
    fig.update_layout(title_text = 'Moving Average plots of all Features')
    fig.write_image('figures/MovingAverage.png', width = 1050, height = 750)

def main():
    df = pd.read_csv('data/master.csv', header = 0, index_col = None)
    feats = df.columns[2:]
    # test = df['Corruption']
    # get_stats(test,'Corruption')
    summaries = []
    for feat in feats:
        summaries.append(get_stats(df,feat))
    summary = pd.DataFrame(data = summaries, columns = ['Feature','Mean','Median','Min',
                                                        'Max','Variance','Missing'])
    # summary.to_csv('results/Summary Statistics.csv', header = True, index = False)
    df = pd.read_csv('data/master_cleaned.csv', header = 0, index_col = None).drop(['Label','Response'],axis=1)
    distance_df = pd.read_csv('data/country_coords.csv', header = 0, index_col = None)
    heatmap(df)
    distancemap(distance_df)
    histograms(df,feats)
    timeseries(df,feats)
    movingavg(df,feats)

print(os.getcwd())
main()
