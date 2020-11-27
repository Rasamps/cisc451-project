import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import scale


def load_data(cd, locations):
    #create a dictionary containing a dataframe of air quality data for each country
    df_dict = {}
    for l in range(0,len(locations)):
        path = cd + "\data\\" + locations[l] + "_data.csv"
        df = pd.read_csv(path, header = 0, index_col = None, usecols=['utc', 'value'])
        df['value'].loc[(df['value'] < 0)] = df['value'].median()
        df_dict[l] = df

    return df_dict

def visualize_data(locations, df_dict):
    #plot the PM2.5 values for each country to get idea of cleaning needed
    #plot histogram of values as well to get idea of cutoff values to use for values that are
    #unreasonably small or large
    for l in range(0,len(locations)):
        df = df_dict[l]
        plt.figure()
        plt.plot(df['value'])
        plt.ylim(0,500)
        plt.show()
        #vals = df[['value']]
        hist = df[['value']].hist(bins=100)

def clean_data(df_dict):
    #based on histograms and time series plots from visualize_data,
    #remove values that are unusually large or small (bias involved)
    #replace these values with median of the data
    indices = [0,1,3,5,6,7,8,9,10,12,13,14,15,16,17,18]
    thresholds = [500,250,150,150,200,250,220,375,200,100,250,250,200,250,250,250]
    k = 0
    for ind in indices:
        df = df_dict[ind]
        df['value'].loc[(df['value'] > thresholds[k])] = df['value'].median()
        k+=1
    return df_dict

def downsample(k, df_dict):
    #down sample time series data to 1500 observations so it is more
    #manageable for clustering algorithm
    #return a 2d numpy array where each row is a countries PM2.5 values
    aq_array = np.zeros((k,1500))
    for l in range(0,k):
        df = df_dict[l]
        sampling_rate = int(len(df)/1500)
        df = df.iloc[::sampling_rate,:]
        val_list = df['value'].tolist()
        val_list = val_list[0:1500]
        aq_array[l,:] = val_list

    return aq_array

def standardize(array, showplot):
    #standardize the data, display plot if showplot is True
    array_stand = scale(array)
    if showplot == True:
        plt.figure()
        for l in range(0,array_stand.shape[0]):
            data = array_stand[l,:]
            plt.plot(data)
        plt.title('Standardized Data')
        #plt.ylim(-2.5,4.5)
        plt.show()
    return array_stand

def run_preprocess(cd, locations):
    df_dict = load_data(cd, locations)
  #  visualize_data(aq_locations, df_dict) #visualize raw data
    df_dict_clean = clean_data(df_dict)
  #  visualize_data(aq_locations, df_dict_clean) #visualize data again after removing erroneous values

    aq_array = downsample(len(locations), df_dict_clean)
    aq_array_stand = standardize(aq_array, True)
    return aq_array_stand
