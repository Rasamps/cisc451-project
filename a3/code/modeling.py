import os
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.cluster import KMeans

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

#Move to the data directory to be able to access and write data.
os.chdir('../data')

def process(rfm,scale):
    rfm = rfm[~((rfm.Frequency > 145.8) | (rfm.Monetary > 726.25) | (rfm.Returns > 7))]
    if (scale):
        copy = pd.DataFrame(data = scaler.fit_transform(rfm[['Recency','Frequency','Monetary','Returns']]),
                            columns = ['Recency','Frequency','Monetary','Returns'])
        copy['CustomerID'], copy['Country'] = rfm.CustomerID, rfm.Country
        return copy
    else:
        return rfm

def elbow_check(rfm):
    pca = PCA(n_components = 2)
    ks = [3,4,5]
    inertias = []
    for k in ks:
        kmodel = KMeans(n_clusters = k).fit(
            pca.fit_transform(rfm[['Recency','Frequency','Monetary','Returns']])
        )
        inertias.append(kmodel.inertia_)
    fig = go.Figure(
        go.Scatter(
            x = ks, y = inertias
        )
    )
    fig.show()

def kmeans(rfm):
    pca_rfm = pd.DataFrame(
        data = PCA(n_components = 2).fit_transform(rfm[['Recency','Frequency','Monetary','Returns']]),
        columns = ['X','Y']
    )
    kmodel = KMeans(n_clusters = 5).fit(pca_rfm)
    rfm['Labels'], pca_rfm['Labels'] = kmodel.labels_, kmodel.labels_
    fig = go.Figure(go.Scatter(
        x = pca_rfm.X, y = pca_rfm.Y,
        mode = 'markers', marker_color = pca_rfm.Labels
    ))
    fig.show()
    return rfm

def distribution(rfm):
    clusters = rfm.Labels.unique()
    clusters.sort()
    percentages = []
    for cluster in clusters:
        percentages.append(rfm.Labels[rfm.Labels == cluster].count() / len(rfm.Labels))
    fig = go.Figure(go.Bar(
        x = [1,2,3,4,5], y = percentages
    ))
    fig.show()

def metrics(rfm_og):
    clusters = rfm_og.Labels.unique()
    clusters = clusters[0:len(clusters)-1]
    clusters.sort()
    attributes = ['Recency','Frequency','Monetary','Returns']
    for attribute in attributes:
        for cluster in clusters:
            print('---------------------------------------------')
            print('The statistics of',attribute,'in cluster',int(cluster+1),' are:')
            print('Mean:',np.mean(rfm_og[attribute][rfm_og.Labels == cluster]))
            print('Max:',max(rfm_og[attribute][rfm_og.Labels == cluster]))
            print('Min:',min(rfm_og[attribute][rfm_og.Labels == cluster]))
            print('Median:',np.median(rfm_og[attribute][rfm_og.Labels == cluster]))
            print('---------------------------------------------')

def wordclouds(purchases):
    clusters = purchases.Labels.unique()
    clusters.sort()
    fig,ax = plt.subplots(5,1, figsize = (10,10))
    for cluster,i in zip(clusters,range(0,5)):
        cluster_text = ' '.join(product for product in purchases.Description[purchases.Labels == cluster])
        wordcloud = WordCloud(max_font_size=50,max_words=25,background_color='black').generate(cluster_text)
        ax[i].imshow(wordcloud, interpolation='bilinear')
        ax[i].set_title('Cluster: ' + str(cluster), fontsize=15)
        ax[i].axis('off')
    plt.show()


def main():
    rfm = pd.read_csv('rfm.csv', header = 0, index_col = None)
    rfm_og = rfm
    rfm = process(rfm,True)
    rfm_og = process(rfm_og,False)
    # elbow_check(rfm)
    rfm = kmeans(rfm)
    # distribution(rfm)
    rfm_og['Labels'] = rfm.Labels
    # metrics(rfm_og)

    purchases = pd.read_csv('purchases.csv', header = 0, index_col = None)
    purchases = purchases.join(rfm.set_index('CustomerID'), on = 'CustomerID', how = 'inner', lsuffix = '_caller', rsuffix = '_other')
    wordclouds(purchases)

main()
