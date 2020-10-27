import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

os.chdir('..') #Makes accessing the data folder more straight-forward.

def main():
    #read in data
    data = pd.read_csv('data/C2T1_Train.csv')

    data_no_null = missing_values(data)

    #remove encounter_rd2 and patient_nbr2
    data = data_no_null.drop(columns = ['encounter_id2', 'patient_nbr2'])

    show_numeric_correlations(data)
    data.to_csv('train_data_cleaned1.csv')

def missing_values(data):
    #{
    # Calculate the percentage of missing values in each column of data
    # Remove columns with more than 40% missing values
    # }


    #replace any '?' values with nan
    data = data.replace({'?': np.nan})
    #get list of columns/features with any nan values
    data_with_any_null = data[data.columns[data.isnull().any()].tolist()]

    #for each of the columns with nans, calculate percentage of nans
    percent_missing = []
    for col in list(data_with_any_null.columns):
        percent = data_with_any_null[col].isnull().sum() * 100/len(data_with_any_null)
        percent_missing.append(percent)

    columns = data_with_any_null.columns.tolist()
    #create dataframe with percentage of missing values for each of those features
    missing_values_df = pd.DataFrame({'Features' : columns,
                                'Percent Missing Values' : percent_missing})

    data_no_null = data
    #remove any columns with more than 40% missing values
    for index, row in missing_values_df.iterrows():
        if row['Percent Missing Values'] > 40:
            column_name = row['Features']
            data_no_null = data_no_null.drop(columns = [column_name])

    return data_no_null

def show_numeric_correlations(data):
    #{
    # Calculate Pearson Correlation Coefficient for all the numeric columns
    # Display the values using a heatmap
    # }
    #get numeric columns
    numerics=['int16','int32', 'int64', 'float16', 'float32', 'float64']
    numeric_data = data.select_dtypes(include=numerics)
    correlations = numeric_data.corr(method='pearson')
    #generate mask for upper triangle of correlation matrix
    mask = np.triu(np.ones_like(correlations, dtype=bool))
    #create colormap
    cmap = sns.diverging_palette(230,20,as_cmap=True)

    plt.figure()
    plot = sns.heatmap(correlations, mask=mask, cmap=cmap, vmax=1, center=0,
    square=True, linewidths=0.5, cbar_kws={"shrink":0.5})
    plt.show()

print(os.getcwd())
# main()
