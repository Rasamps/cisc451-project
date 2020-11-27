import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def build_model(i, train_indices, pm25, df):
    #buld a linear regression model using a single country's data
    #return the model
    idx = train_indices[i]
    X = df.iloc[idx:idx+19].values
    Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
    m = LinearRegression(fit_intercept=True).fit(X,Y)

    return m

def fit_models(i, test_indices, pm25, df, models):
    #predict a single country's air quality data using each of the 13 models
    #calculate the final prediction values by averaging the results from all 13 models
    #return both the actual air quality data and the predicted air quality data

    idx = test_indices[i]
    Y_actual = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
    X = df.iloc[idx:idx+19].values
    all_predictions = np.zeros((len(models),len(Y_actual)))
    print(all_predictions.shape)
    for i in range(0,len(models)):
        m = models[i]
        Y_pred = m.predict(X)
        all_predictions[i,:] = Y_pred

    Y_pred_avg = np.mean(all_predictions, axis=0)

    df=pd.DataFrame({'Actual':Y_actual, 'Predicted':Y_pred})

    return Y_actual, Y_pred

def model_driver(test_indices, pm25, df, models):
    #calls fit_models for each of the test countries
    #saves the predictions in a single numpy array so each row contains predictions for a single country

    all_Y_actual = np.zeros((len(test_indices),19))
    all_Y_pred = np.zeros((len(test_indices),19))

    for i in range(0,len(test_indices)):
        Y_actual, Y_pred = fit_models(i, test_indices, pm25, df, models)
        all_Y_actual[i,:] = Y_actual
        all_Y_pred[i,:] = Y_pred

    return all_Y_actual, all_Y_pred


def model_evaluation(all_Y_actual, all_Y_pred):
    #calculate rmse for each of the test countries
    rms1 = np.sqrt(mean_squared_error(all_Y_actual[0,:],all_Y_pred[0,:]))
    rms2 = np.sqrt(mean_squared_error(all_Y_actual[1,:],all_Y_pred[1,:]))
    rms3 = np.sqrt(mean_squared_error(all_Y_actual[2,:],all_Y_pred[2,:]))
    rms4 = np.sqrt(mean_squared_error(all_Y_actual[3,:],all_Y_pred[3,:]))
    rms5 = np.sqrt(mean_squared_error(all_Y_actual[4,:],all_Y_pred[4,:]))
    rms6 = np.sqrt(mean_squared_error(all_Y_actual[5,:],all_Y_pred[5,:]))

    #print(all_Y_actual)
    #print(all_Y_pred)
    print('RMSE:')
    print(rms1)
    print(rms2)
    print(rms3)
    print(rms4)
    print(rms5)
    print(rms6)
