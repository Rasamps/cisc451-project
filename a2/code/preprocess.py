import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def missing(data,labels):
    for lbl in labels:
        imp = SimpleImputer(missing_values = '?', strategy = 'most_frequent')
        data[lbl] = imp.fit_transform(data[lbl].to_numpy().reshape(-1,1))
    return data

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
