import os
import pandas as pd
import numpy as np
import datetime as datetime

def feature_average(df,feat):
    means = []
    for year in df.Year.unique():
        means.append(np.nanmean(df[feat][df.Year == year].to_numpy('float32')))
    years = pd.DataFrame(data = [[y,m] for y,m in zip(df.Year.unique(),means)], columns = ['Year','Mean'])
    years.loc[:,'Mean'] = just_replace(years[['Year', 'Mean']],'Mean')
    years = years['Mean'].to_numpy('float32')
    return years

def get_averages():
    df = pd.read_csv('data/master.csv', header = 0, index_col = None)
    df = df.replace('..', np.nan)
    features = {}
    for feat in df.columns[2:]:
        features[feat] = feature_average(df[['Year',feat]],feat)
    return features

def replace_all(feat):
    return averages[feat]

def replace_front(df,feat):
    df.loc[:,'Year'] = pd.to_datetime(df['Year'], format = '%Y')
    df.set_index('Year', inplace = True)
    df.loc[:,feat] = df.loc[:,feat].astype('float32')
    interp = df.iloc[::-1].interpolate(method = 'time')[::-1].to_numpy()
    return interp

def just_replace(df,feat):
    df.loc[:,'Year'] = pd.to_datetime(df['Year'], format = '%Y')
    df.set_index('Year', inplace = True)
    df.loc[:,feat] = df.loc[:,feat].astype('float32')
    interp = df.interpolate(method = 'time').to_numpy()
    return interp

def fill_missing(df,country):
    for feat in df.columns[2:]:
        check = df[feat].astype('float32')
        if (check.isnull().all() == True):
            df.loc[:,feat] = replace_all(feat)
        elif (np.isnan(check.iloc[0]) == True):
            df.loc[:,feat] = replace_front(df[['Year', feat]],feat)
        else:
            df.loc[:,feat] = just_replace(df[['Year', feat]],feat)
    return df

def main():
    df = pd.read_csv('data/master.csv', header = 0, index_col = None)
    df = df.replace('..', np.nan)
    countries = []
    for country in df.Country.unique():
        countries.append(fill_missing(df[df.Country == country], country))
    final = countries[0]
    for df_country in countries[1:]:
        final = final.append(df_country, ignore_index = True)
    final.to_csv('data/master_cleaned.csv', header = True, index = False)
    # test = fill_missing(df[df.Country == 'Uganda'], 'Uganda')

print(os.getcwd())
averages = get_averages()
main()
