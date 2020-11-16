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
    fig.update_xaxes(title_text = 'K')
    fig.update_yaxes(title_text = 'Sum of squared distances of records to their closest cluster center')
    fig.update_layout(title_text = 'Elbow Method Test')
    fig.show()

def kmeans(rfm):
    pca_rfm = pd.DataFrame(
        data = PCA(n_components = 2).fit_transform(rfm[['Recency','Frequency','Monetary','Returns']]),
        columns = ['X','Y']
    )
    kmodel = KMeans(n_clusters = 5).fit(pca_rfm)
    distances = pd.DataFrame(data = kmodel.transform(pca_rfm), columns = ['Cluster1','Cluster2','Cluster3','Cluster4','Cluster5'])
    rfm['Labels'], pca_rfm['Labels'], distances['Labels'] = kmodel.labels_, kmodel.labels_, kmodel.labels_
    fig = go.Figure(go.Scatter(
        x = pca_rfm.X, y = pca_rfm.Y,
        mode = 'markers', marker = dict(color=pca_rfm.Labels, size = 5, showscale=True)
    ))
    fig.update_xaxes(title_text = 'Principal Component 1')
    fig.update_yaxes(title_text = 'Principal Component 2')
    fig.update_layout(title_text = 'K-Means Clustering Results')
    fig.show()
    print('The K-means clustering metrics are...\n')
    for cluster in range(1,6):
        curr = 'Cluster'+str(cluster)
        s = np.sum((distances[curr][distances.Labels == (cluster-1)])**2)
        print('For Cluster',cluster,'the ssd of the points is:',s)

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
    fig.update_yaxes(title_text = 'Percentage')
    fig.update_xaxes(title_text = 'Cluster')
    fig.update_layout(title_text = 'Distribution of customers in the five clusters')
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
        ax[i].axis('off')
    plt.show()


def main():
    rfm = pd.read_csv('rfm.csv', header = 0, index_col = None)
    rfm_og = rfm
    rfm = process(rfm,True)
    rfm_og = process(rfm_og,False)
    elbow_check(rfm)
    rfm = kmeans(rfm)
    distribution(rfm)
    rfm_og['Labels'] = rfm.Labels
    metrics(rfm_og)

    purchases = pd.read_csv('purchases.csv', header = 0, index_col = None)
    purchases = purchases.join(rfm.set_index('CustomerID'), on = 'CustomerID', how = 'inner', lsuffix = '_caller', rsuffix = '_other')
    wordclouds(purchases)

main()
