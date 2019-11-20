#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cv', nargs='?', const=3, type=int,
                    help='do cross validation with specified number of folds')
args = parser.parse_args()

# Only initialize dask-mpi and dask if performing cross validation:
if args.cv:
    from dask_mpi import initialize
    initialize()

    from distributed import Client
    client = Client()

    from dask_ml.model_selection import GridSearchCV, train_test_split
else:
    from sklearn.model_selection import GridSearchCV, train_test_split

from azureml.core import Run
import numpy as np
from sklearn.datasets import load_digits
import sklearn.metrics as skmetrics
from sklearn.svm import SVC

run = Run.get_context()

# For a bigger dataset, try
# x, y = make_classification(5000, n_classes=16, n_informative=13)
digits = load_digits()
x = digits.data
y = digits.target

# Obtain cross validation performance for single parameter setting:
param_space = {
    'C': [1]
}

model = SVC(kernel='rbf')
if args.cv:
    search = GridSearchCV(model, param_space, scoring=['accuracy'],
                          refit=False, cv=args.cv)
    search.fit(x, y)
    run.log('accuracy_mean', search.cv_results_['mean_test_accuracy'][0])
    run.log('accuracy_std', search.cv_results_['std_test_accuracy'][0])
else:
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    run.log('accuracy', skmetrics.accuracy_score(y_test, y_test_pred))
