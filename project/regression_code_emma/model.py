#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#%%
    df = pd.read_csv('master_with_aq.csv', header = 0)
    df = df.drop(['Unnamed: 0', 'Label', 'Country'], axis=1)
    #df = StandardScaler().fit_transform(df)
#%%
df[df.columns] = StandardScaler().fit_transform(df)
#%%
df
#%%
df = pd.read_csv('master_with_aq.csv', header = 0)
pm25 = df['PM2.5']
df = df.drop(['Unnamed: 0', 'PM2.5', 'Label', 'Country'], axis=1)
#%%
nums = np.arange(0,360,19)

train_indices = np.random.choice(nums, 13, replace=False)
train_indices

# %%
#MODEL 1
idx = train_indices[0]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m1 = LinearRegression(fit_intercept=True).fit(X,Y)


# %%
#MODEL 2
idx = train_indices[1]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m2 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 3
idx = train_indices[2]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m3 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 4
idx = train_indices[3]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m4 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 5
idx = train_indices[4]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m5 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 6
idx = train_indices[5]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m6 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 7
idx = train_indices[6]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m7 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 8
idx = train_indices[7]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m8 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 9
idx = train_indices[8]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m9 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 10
idx = train_indices[9]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m10 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 11
idx = train_indices[10]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m11 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 12
idx = train_indices[11]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m12 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
#MODEL 13
idx = train_indices[12]
X = df.iloc[idx:idx+19].values #.values[1:]
Y = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
m13 = LinearRegression(fit_intercept=True).fit(X,Y)
# %%
test_indices = np.setdiff1d(nums, np.sort(train_indices))
test_indices
# %%
idx = test_indices[0]
Y_new = pd.Series.to_numpy(pm25.iloc[idx:idx+19])
X_new = df.iloc[idx:idx+19].values
Y_pred = m1.predict(X_new)
#rms2 = np.sqrt(mean_squared_error(Y2,Y2_pred))


# %%
Y_pred
# %%
import pandas as pd
import numpy as np
#import modeling as md
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('master_with_aq.csv', header = 0)
df = df.drop(['Unnamed: 0', 'Label', 'Country'], axis=1)
df[df.columns] = StandardScaler().fit_transform(df)
#pm25 = df['PM2.5']
df2 = df.drop(['PM2.5'], axis=1)

#%%
mod = sm.tsa.statespace.SARIMAX(df['PM2.5'], df2)
res = mod.fit(disp=False)
print(res.summary())
# %%
#
mod = sm.tsa.statespace.ARIMAX(df['PM2.5'], df2)
res = mod.fit(disp=False)
print(res.summary())
# %%
#df2 = df2.drop(['CO2','DrinkingWater','ForeignInvest','EmployRatio','AirFreight'], axis=1)
df2 = df2.drop(['ForestArea','Rainfall', 'Sanitation'], axis=1)
model = sm.OLS(df['PM2.5'], df2).fit()
predictions = model.predict(df2) 

print_model = model.summary()
print(print_model)
## %%

# %%
