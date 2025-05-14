"""
The main utility functions. List goes like this:
- classify_features
# - timef
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from time import perf_counter


def classify_features(df):
    """
    Splits features into numerical and categorical features. Depends on Pandas library

    Parameters
    ----------
    DataFrame: Pandas DataFrame

    Return
    ------
    numerical_features, categorical_features
        Two list composed of given features
    """
    all_features = df.columns.to_list()
    numerical_features = df.describe().columns.to_list()
    categorical_features = df.select_dtypes(exclude="number").columns.to_list()
    return numerical_features, categorical_features, all_features


def timef(obj, *args):
    """
    Time a function or class operations with perf_counter

    Parameters
    ----------
    f: function or object performing an action
    *args: The arguments to be passed to given function
    """

    def wrap_function(*args, **kwargs):
        start = perf_counter()
        result = obj(*args, **kwargs)
        finish = perf_counter()
        print(f"Operation {obj.__name__!r} executed in {(finish-start):.4f}s")

        return result

    return wrap_function(*args)


def draw_decision_boundary(estimator, X, y):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.5), np.arange(x2_min, x2_max, 0.5)
    )

    Z = estimator.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=4)
    plt.title("Decision Boundaries")
    plt.show()
