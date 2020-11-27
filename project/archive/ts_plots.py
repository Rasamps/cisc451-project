import plotly.express as px
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import pandas as pd
print(os.getcwd())

locations = ['bd','bh','ca','fr','gt','hk','hu','id','in','iq','kw','lk','lu','mn','no',
            'np','pe','pk','ug','vm']

for l in range(0,len(locations)):
    fig, ax = plt.subplots()
    path = "data/"+ locations[l] +"_data.csv"
    df = pd.read_csv(path, header = 0, index_col = None)
    #----------Descriptive Statistics-------------
    print("Current Country is: " + locations[l])
    print(np.min(df.value))
    print(np.max(df.value))
    print(np.mean(df.value))
    print(np.var(df.value))
    #-----------------Plotting--------------------
    ax.plot(df.utc,df.value)
    plt.xlabel("Date")
    plt.ylabel("Concentration of PM 2.5")
    plt.title("PM 2.5 Time Series data for: " + locations[l])
    plt.show()
