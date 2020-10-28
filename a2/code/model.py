import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def sup_vec(X_train, y_train, X_test):
    params = {'C': [0.1,1.0,10],
              'kernel': ['rbf','poly']
              'degree': [3,4],
              'gamma': ['scale','auto'],
              'coef0': [0.0,0.5]
    }
    svc = SVC()
    crossval = GridSearchCV(svc, params, cv = 3, scoring = 'accuracy', verbose = 2)
    crossval.fit(X_train, y_train)
    print("The best parameter combination was : ", crossval.best_params_)
    val_preds = crossval.predict(X_train)
    print(accuracy_score(y_train, val_preds))
    print(confusion_matrix(y_train, val_preds))
    return crossval.predict(X_test)

def nn(X_train, y_train, X_test):
    params = {
        'hidden_layer_sizes': [(25,25,25),(10,10),(50)],
        'activation': ['relu','tanh'],
        'alpha': [0.0001, 0.01, 1.0]
    }
    mlp = MLPClassifier()
    crossval = GridSearchCV(mlp, params, cv = 3, scoring = 'accuracy')
    crossval.fit(X_train, y_train)
    print(confusion_matrix(y_train, crossval.predict(X_train)))
    return crossval.predict(X_test)
