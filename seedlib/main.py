"""
    Random Useful Snippets for doing stuff in python
    Ankur Goswami
"""

import pandas as pd
from math import sqrt, e, pi
from numpy import linalg, cov


def cat(filename):
    with open(filename, 'r') as rf:
        for line in rf:
            print(line)


def wcl(filename, without_empty_line=True):
    with open(filename, 'r') as rf:
        count = 0
        for line in rf:
            if without_empty_line:
                if line == "":
                    continue
            count += 1
        return count


def pd_print(df, jupyter=True):
    with pd.option_context('display.max_rows', 4000, 'display.max_columns', 4000, 'display.max_colwidth', 4000):
        if jupyter:
            display(df)
        else:
            print(df)


def split_data(df, train_size=0.8):
    train = df.sample(train_size)
    test = df.loc[~df.index.isin(train.index)]
    return train, test


def get_n_sets(df, n=10, train_size=0.8):
    train_sets = test_sets = []
    for i in range(0, n):
        train, test = split_data(df, train_size)
        train_sets.append(train)
        test_sets.append(test)
    return zip(train_sets, test_sets)


def gaussian_kernel(sigma, X, sample_X):
    return e ** (-linalg.norm(X - sample_X) ** 2 / (2 * sigma ** 2)) / (sqrt(2 * pi * sigma ** 2))


def find_amise_optimal_sigma(covariance_diagonals, n, d=1):
    if type(covariance_diagonals) != list:  # 0d case
        covariance_diagonals = [covariance_diagonals]
    sum_covariance = sum(covariance_diagonals)
    sigma = sqrt(sum_covariance / d)
    return sigma * (4 / ((2*d + 1)*n))**(1/(d+4))


def parzen_window_gaussian(X, sample_X):
    n = len(sample_X)
    results = []
    optimal_sigma = find_amise_optimal_sigma(cov(sample_X), n)
    for x in X:
        sum = 0
        for sample_x in sample_X:
            sum += gaussian_kernel(optimal_sigma, x, sample_x)
        results.append(sum)
    return [result / n for result in results]
