#!/usr/bin/env python3

from dask_mpi import initialize
initialize()

from distributed import Client
client = Client()

import argparse

from azureml.core import Run

import numpy as np
from sklearn.datasets import load_digits
from dask_ml.model_selection import GridSearchCV
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument('--cv', nargs='?', const=3, type=int,
                    help='do cross validation with specified number of folds')
args = parser.parse_args()

run = Run.get_context()
# For a bigger dataset, try
# x, y = make_classification(5000, n_classes=16, n_informative=13)
digits = load_digits()
x = digits.data
y = digits.target

param_space = {
    'C': [1]
}

model = SVC(kernel='rbf')
search = GridSearchCV(model, param_space, scoring=['accuracy'],
                      refit=False, cv=args.cv)
search.fit(x, y)
run.log('accuracy_mean', search.cv_results_['mean_test_accuracy'][0])
run.log('accuracy_std', search.cv_results_['std_test_accuracy'][0])
