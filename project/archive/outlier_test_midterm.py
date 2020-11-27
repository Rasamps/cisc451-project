import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr

locations = ['bd','bh','ca','gt','hk','hu','id','in','iq','kw','lk','lu','mn','no',
            'np','pe','pk','ug','vm']

def outlier_ansys(df):
    neg_records = df.value[df.value < 0].count()
    over_records = df.value[df.value > 200].count()
    expected_values = df[(df.value < 200) & (df.value > 0)]
    iqr_rng = iqr(expected_values.value)
    outliers = expected_values.value[expected_values.value > iqr_rng*1.5].count()
    return [neg_records, over_records, outliers]

def plot_outliers(results, df):
    fig,ax = plt.subplots()
    x = np.arange(3)
    print(results)
    # print(df.value)
    plt.bar(x,results)
    plt.xticks(x,("Negative Misrecordings", "Overtly High Recordings", "IQR Outliers"))
    plt.show()

def main():
    for ctry in locations:
        path = "data/"+ ctry +"_data.csv"
        df = pd.read_csv(path, header = 0, index_col = None)
        results = outlier_ansys(df)
        plot_outliers(results,df)

main()
