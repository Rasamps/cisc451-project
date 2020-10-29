import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import scipy.stats as ss
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

os.chdir('..') #Makes accessing the data folder more straight-forward.

def main():
    #read in data
    data = pd.read_csv('data/C2T1_Train.csv') #Read in the data.
    data = missing_values(data) #Drop features which over 40% of the data missing.
    #Remove encounter_rd2 and patient_nbr2
    data = data.drop(columns = ['encounter_id2', 'patient_nbr2']) #These are removed as they are the unique identifiers and shouldn't be considered for the analysis
    show_numeric_correlations(data) #Create plots showing the correlation between numeric features
    mi_select(data)
    # data.to_csv('train_data_cleaned1.csv')

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

def mi_select(data):
    #Separate into input features and target variable.
    X, y = data.loc[:, data.columns != 'readmitted'], data['readmitted']

    #Encode categorical variables
    categorical_cols = X.select_dtypes(['object']).columns
    numeric_cols = X.select_dtypes(['int64']).columns
    X_num, X_cat = X[numeric_cols], X[categorical_cols]
    X_cat[categorical_cols] = X_cat[categorical_cols].astype('category')
    X_cat[categorical_cols] = X_cat[categorical_cols].apply(lambda elem: elem.cat.codes)

    def get_ratio(col):
        vals = col.unique()
        ratio = 0
        for val in vals:
            curr = col[col == val].count() / len(col)
            if (curr > ratio):
                ratio = curr
        return ratio

    ratios = []
    for col in categorical_cols:
        ratios.append(get_ratio(X[col]))

    mi_num = mutual_info_classif(X_num,y, discrete_features = 'auto')
    mi_cat = mutual_info_classif(X_num,y, discrete_features = 'auto')

    fig_num = go.Figure([go.Bar(x = numeric_cols, y = mi_num, name = "Mutual Info")])
    fig_num.show()

    fig_cat = go.Figure([go.Bar(x = categorical_cols, y = mi_cat, name = "Mutual Info"),
                        go.Scatter(x = categorical_cols, y = ratios, name = "Label Ratio",
                                    mode = 'lines+markers')])
    fig_cat.show()

main()
