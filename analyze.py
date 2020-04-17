import numpy as np
import pandas as pd

import scipy.stats as ss
import scipy.special as sp

import matplotlib.pyplot as plt
import seaborn as sns


## Basic initial stats.

def get_basic_stats(df, datecol):
    '''
    :input pandas.dataframe df:
    :input String datecol:
    '''
    # number of rows
    num_rows = len(df)
    print('total # of rows: {} \n'.format(num_rows))

    # number of columns
    num_cols = len(df.columns)
    print('total # of columns: {} \n'.format(num_cols))
    print(list(df.columns))

    # number of unique items #TODO

    # date range for data
    min_date = min(df[datecol])
    max_date = max(df[datecol])
    print(min_date)
    print(max_date)

    # geos covered #TODO

    return (num_rows, num_cols)

## Check coverage for the dataset.

def report_overall_coverage(df):
    '''
    :input pandas.dataframe df:
    '''
    return (100*df.isna().sum().sum() / float(len(df)*len(df.columns)))


def report_column_coverage(col):
    '''
    :input numpy.ndarray col:
    '''
    count_missing_vals = col.isna().sum()
    print('# missing values:', count_missing_vals)
    
    percent_missing = 100*count_missing_vals / float(len(col))
    print('% missing values:', percent_missing)
    
    print('# values:', len(col))
    
    return count_missing_vals, percent_missing, len(col)


## Determining outliers.

# Boxplot method. Why? Z-score biased. Other methods? DBScan, Isolation forests.
def plot_outliers(col, title='', xlabel=''):
    plt.figure(figsize=[16,4])
    sns.boxplot(col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()
    
    
def find_outliers(col):
    q1, q3 = np.percentile(col.dropna(), [25,75])
    iqr = q3 - q1

    lower_bound = q1 - (1.5*iqr) 
    upper_bound = q3 + (1.5*iqr)

    outliers_indices = np.where(col.dropna() > upper_bound)[0]  # drop missing values
    print('# outliers:', len(outliers_indices))    
    percent_outliers = 100*len(outliers_indices) / float(len(col.dropna()))
    print('% outliers:', percent_outliers)
    
    return outliers_indices, len(outliers_indices), percent_outliers

