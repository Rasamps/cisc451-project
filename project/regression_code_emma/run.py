import pandas as pd
import numpy as np
import modeling as md
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def main():
    df = pd.read_csv('master_with_aq.csv', header = 0)
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
    
    all_Y_actual = np.zeros((len(test_indices),19))
    all_Y_pred = np.zeros((len(test_indices),19))
    for i in range(0,len(test_indices)):
        Y_actual, Y_pred = md.fit_models(i, test_indices, pm25, df, models)
        all_Y_actual[i,:] = Y_actual
        all_Y_pred[i,:] = Y_pred
    
    rms1 = np.sqrt(mean_squared_error(all_Y_actual[0,:],all_Y_pred[0,:]))
    rms2 = np.sqrt(mean_squared_error(all_Y_actual[1,:],all_Y_pred[1,:]))
    rms3 = np.sqrt(mean_squared_error(all_Y_actual[2,:],all_Y_pred[2,:]))
    rms4 = np.sqrt(mean_squared_error(all_Y_actual[3,:],all_Y_pred[3,:]))
    rms5 = np.sqrt(mean_squared_error(all_Y_actual[4,:],all_Y_pred[4,:]))
    rms6 = np.sqrt(mean_squared_error(all_Y_actual[5,:],all_Y_pred[5,:]))

    #print(Y_actual)
    #print(Y_pred)
    print('Linear Regression:')
    print(rms1)
    print(rms2)
    print(rms3)
    print(rms4)
    print(rms5)
    print(rms6)
    #calculate rmse for each 

 
main()