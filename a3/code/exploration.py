import os
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from scipy.stats import iqr

import plotly.graph_objects as go
from plotly.subplots import make_subplots

#Move to the data directory to be able to access and write data.
os.chdir('../data')

def feat_plot(df):
    fig = make_subplots(rows = 2, cols = 2)
    fig.add_trace(go.Histogram(
        x = df.Recency),
        row = 1, col = 1)
    fig.add_trace(go.Histogram(
        x = df.Frequency),
        row = 1, col = 2)
    fig.add_trace(go.Histogram(
        x = df.Monetary),
        row = 2, col = 1)
    fig.add_trace(go.Histogram(
        x = df.Returns),
        row = 2, col = 2)
    fig.show()

def pca_plot(df):
    pca = PCA(n_components = 2)
    w_returns = pca.fit_transform(scaler.fit_transform(df[['Recency','Frequency','Monetary','Returns']]))
    wo_returns = pca.fit_transform(scaler.fit_transform(df[['Recency','Frequency','Monetary']]))

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(
        x = [xi[0] for xi in w_returns], y = [yi[1] for yi in w_returns],
        mode = 'markers'),
        row = 1, col = 1
    )
    fig.add_trace(go.Scatter(
        x = [xi[0] for xi in wo_returns], y = [yi[1] for yi in wo_returns],
        mode = 'markers'),
        row = 1, col = 2
    )
    fig.show()

def iqr_outliers(df):
    for col in ['Recency','Frequency','Monetary','Returns']:
        print('The IQR for', col, 'is: ')
        curr_iqr = iqr(df[col], axis = 0)
        print(curr_iqr*3)
        print('The number of values above the 75th percentile are: ')
        print(df[col][df[col] > curr_iqr].count())

def main():
    rfm = pd.read_csv('rfm.csv', header = 0, index_col = None)
    feat_plot(rfm)
    pca_plot(rfm)
    iqr_outliers(rfm)
    print('Originally we had...',rfm.shape,'records.')
    rfm = rfm[~((rfm.Frequency > 145.8) | (rfm.Monetary > 726.25) | (rfm.Returns > 7))]
    print('Now we have:',rfm.shape)
    feat_plot(rfm)
    pca_plot(rfm)
    print('With outliers removed we can now plot the scaled features...')
    rfm = pd.DataFrame(data = scaler.fit_transform(rfm[['Recency','Frequency','Monetary','Returns']]),
                        columns = ['Recency','Frequency','Monetary','Returns'])
    feat_plot(rfm)

main()
