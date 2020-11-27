from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

def build_model(i, train_indices, pm25, df):
    idx = train_indices[i]
    X = df.iloc[idx:idx+19].values
    Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
    m = LinearRegression(fit_intercept=True).fit(X,Y)

    #m = DecisionTreeRegressor(criterion='mse',max_depth=4).fit(X,Y)

    return m


def fit_models(i, test_indices, pm25, df, models):
    idx = test_indices[i]
    Y_actual = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
    X = df.iloc[idx:idx+19].values
    all_predictions = np.zeros((len(test_indices),len(Y_actual)))

    for i in range(0,len(models)):
        m = models[i]
        Y_pred = m.predict(X)
        all_predictions[i,:] = Y_pred

    Y_pred_avg = np.mean(all_predictions, axis=0)

    df=pd.DataFrame({'Actual':Y_actual, 'Predicted':Y_pred})
    print(df)
    return Y_actual, Y_pred
    
        