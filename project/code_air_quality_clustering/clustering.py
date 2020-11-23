import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster, ward, dendrogram
import os


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
     # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix

def hierarchical_clustering(aq_array):
    clf = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(aq_array)
    plt.figure(figsize = (14,6))
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the dendrogram
    Z = plot_dendrogram(clf, p=5, color_threshold = 110)
    plt.show()
    # extract clusters from dendogram (pick 2 clusters based on dendrogram)
    clusters = fcluster(Z, 2, criterion='maxclust')
    # create dataframe where index is the cluster the row belongs to
    aq_clustered = pd.DataFrame(aq_array)
    aq_clustered["cluster"] = clusters
    aq_clustered = aq_clustered.set_index("cluster".split())

    return aq_clustered

def get_clusters(df_clustered):
    #separate a single dataframe into three dataframes by cluster
    

    return cluster1, cluster2 #, cluster3

        

def cluster_metrics(cluster):  
    #get min, max, median, and mean of each country in a cluster

    c_max = cluster.max(axis=1)
    c_min = cluster.min(axis=1)
    c_median = cluster.median(axis=1)
    c_mean = cluster.mean(axis=1)
    return c_max, c_min, c_median, c_mean

def plot_metric(c1_metric, c2_metric, metric_name):
    fig, ax = plt.subplots()
    ax.plot(range(0,len(c1_metric)),c1_metric.tolist(), '-ok', color='r', label='Cluster 1')
    ax.plot(range(0,len(c2_metric)),c2_metric.tolist(), '-ok', color='b', label='Cluster 2')
    #ax.plot(range(0,len(c3_metric)),c3_metric.tolist(), '-ok', color='g', label='Cluster 3')
    #ax.axis('equal')
    plt.title('Cluster ' + metric_name + ' values')
    leg = ax.legend()
    plt.show()

def print_cluster_members(df_clustered, l1, l2, locations):
    c1_countries = [None]*l1
    c2_countries = [None]*l2
    #c3_countries = [None]*l3
    i1 = 0
    i2 = 0
    i3 = 0
    for i in range(0, len(df_clustered)):   
        if df_clustered.index[i] == 1:
            c1_countries[i1] = locations[i]
            i1 += 1
        else: # df_clustered.index[i]  == 2:
            c2_countries[i2] = locations[i]
            i2 += 1
    print('Cluster 1 countries:')
    print(c1_countries)
    print('Cluster 2 countries:')
    print(c2_countries)
    #print('Cluster 3 countries:')
    #print(c3_countries)
    
    return c1_countries, c2_countries #, c3_countries

def visualize_clusters(df, cluster):
    plt.figure()
    for l in range(0,df.shape[0]):
        data = df.iloc[l]
        plt.plot(data[0:200])
    plt.title(cluster)
    plt.ylim(-2.5,4.5)
    plt.show()   

def run_clustering(array_stand):
    aq_clustered = hierarchical_clustering(array_stand) #dataframe with cluster column
    cluster1 = aq_clustered.loc[1] #dataframe containing cluster 1 countries
    cluster2 = aq_clustered.loc[2] #dataframe containing cluster 2 countries
    c1_max, c1_min, c1_median, c1_mean = cluster_metrics(cluster1)
    c2_max, c2_min, c2_median, c2_mean = cluster_metrics(cluster2)

    #plot the min, max, mean, median, values of each of the countries in each cluster
    plot_metric(c1_min, c2_min, 'minimum')
    plot_metric(c1_max, c2_max, 'maximum')
    plot_metric(c1_mean, c2_mean, 'mean')
    plot_metric(c1_median, c2_median, 'median')
    #print the countries that belong to each cluster
    c1_countries, c2_countries = print_cluster_members(aq_clustered, cluster1.shape[0], cluster2.shape[0], aq_locations)

    #compare the data of countries from both clusters
    visualize_clusters(cluster1, 'Cluster 1')
    visualize_clusters(cluster2, 'Cluster 2')

