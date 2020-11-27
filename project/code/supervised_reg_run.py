import os
import pandas as pd
import numpy as np
import modeling as md
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv('data/master_with_aq.csv', header = 0)
    df = df.drop(['Unnamed: 0', 'Label', 'Country'], axis=1)
    df[df.columns] = StandardScaler().fit_transform(df)
    print(df.head())
    pm25 = df['PM2.5']
    df = df.drop(['PM2.5'], axis=1)
    nums = np.arange(0,360,19) #indices of where each new country starts in df

    train_indices = np.random.choice(nums, 13, replace=False) #randomly select 13 countries for training

    m1 = md.build_model(0,train_indices,pm25,df)
    m2 = md.build_model(1,train_indices,pm25,df)
    m3 = md.build_model(2,train_indices,pm25,df)
    m4 = md.build_model(3,train_indices,pm25,df)
    m5 = md.build_model(4,train_indices,pm25,df)
    m6 = md.build_model(5,train_indices,pm25,df)
    m7 = md.build_model(6,train_indices,pm25,df)
    m8 = md.build_model(7,train_indices,pm25,df)
    m9 = md.build_model(8,train_indices,pm25,df)
    m10 = md.build_model(9,train_indices,pm25,df)
    m11 = md.build_model(10,train_indices,pm25,df)
    m12 = md.build_model(11,train_indices,pm25,df)
    m13 = md.build_model(12,train_indices,pm25,df)

    models = [m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13]
    #get indices of countries not in training set to use for test set
    test_indices = np.setdiff1d(nums, np.sort(train_indices))

    all_Y_actual, all_Y_pred = md.model_driver(test_indices, pm25, df, models)



    #calculate rmse to evaluate the quality of each of the models predictions
    md.model_evaluation(all_Y_actual, all_Y_pred)



print(os.getcwd(),'\n')
main()
