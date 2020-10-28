import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

def label(data,labels):
    for lbl in labels:
        le = LabelEncoder()
        data[lbl] = le.fit_transform(data[lbl])
    return data

def one_hot(data,labels):
    ct = ColumnTransformer([('encoder', OneHotEncoder(), labels)],
                            remainder = 'passthrough')
    data = np.array(ct.fit_transform(data))
    return data
