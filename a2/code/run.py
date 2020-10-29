import os
import numpy as np
import pandas as pd
import preprocess
import model

os.chdir('..')

def main():
    feats = ['discharge_disposition_id','admission_source_id','number_outpatient','number_emergency','number_inpatient','number_diagnoses',
            'race','gender','age','max_glu_serum','A1Cresult','metformin','repaglinide','nateglinide','insulin','change','diabetesMed']
    to_label = ['race','gender','age','max_glu_serum','A1Cresult','metformin','repaglinide','nateglinide','insulin','change','diabetesMed']
    ohe = ['race','gender']

    data_train = pd.read_csv('data/C2T1_Train.csv')
    data_test = pd.read_csv('data/C2T1_Test.csv')

    y_train = data_train['readmitted']

    X_train = data_train[feats]
    X_test = data_test[feats]

    X_train = preprocess.missing(X_train,ohe)
    X_test = preprocess.missing(X_test,ohe)

    X_train = preprocess.label(X_train,to_label)
    X_train = preprocess.one_hot(X_train,ohe)
    X_test = preprocess.label(X_test,to_label)
    X_test = preprocess.one_hot(X_test,ohe)

    predictions = model.nn(X_train,y_train,X_test)

    test_labeled = data_test[['encounter_id','patient_nbr']]
    test_labeled['readmitted'] = predictions
    test_labeled.to_csv('C2T1_Test_Labeled.csv', index = False)

main()
