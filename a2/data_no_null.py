import pandas as pd 
import numpy as np
from pandas import io

#read in data
data = pd.read_csv('C2T1Data/C2T1_Train.csv')
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


data_no_null.to_csv('train_data_no_null.csv')
