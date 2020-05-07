#!/usr/bin/env python3

import numpy as np
import pandas as pd

import scipy.stats as ss
import scipy.special as sp

import matplotlib.pyplot as plt
import seaborn as sns


## Basic initial stats.

def get_basic_stats(df, datecol):
    """
    :input pandas.dataframe df:
    :input String datecol:
    """
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
    """
    :input pandas.dataframe df:
    """
    return (100*df.isna().sum().sum() / float(len(df)*len(df.columns)))


def report_column_coverage(col):
    """Report missing values.

    Args:
        col (numpy.ndarray): Column of dataset.
        
    """
    count_missing_vals = col.isna().sum()
    print('# missing values:', count_missing_vals)
    
    percent_missing = 100*count_missing_vals / float(len(col))
    print('% missing values:', percent_missing)
    
    print('# values:', len(col))
    
    return count_missing_vals, percent_missing, len(col)


## Determining outliers.

# Boxplot method. Why? Z-score biased. Other methods? DBScan, Isolation forests.
def plot_outliers(col, title='', xlabel=''):
    """Visualizes outliers with a boxplot.
    
    Args:
        col (numpy.ndarray): Column of dataset.
        title (String): Title to use for plot.
        xlabel (String): Label to use for x-axis.
    
    Returns:
        Boxplot of data in col.
    """
    plt.figure(figsize=[16,4])
    sns.boxplot(col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()
    
    
def find_outliers(col, multiplier=1.5, lower_percentile=25, upper_percentile=75):
    """Finds outliers using the interquartile range.
    
    Args:
        col (numpy.ndarray): Column of dataset.
        multiplier (int): Defaults to 1.5, industry-standard for outliers.
        lower_percentile (int): Default is second quartile a.k. 25th.
        upper_percentile (int): Default is third quartile a.k. 75th.
    
    Returns:
        triple: The indices of the outliers, number of outliers, percent that are outliers.
    """
    q1, q3 = np.percentile(col.dropna(), [lower_percentile, upper_percentile])
    iqr = q3 - q1

    lower_bound = q1 - (multiplier*iqr) 
    upper_bound = q3 + (multiplier*iqr)

    outliers_indices = np.where(col.dropna() > upper_bound)[0]  # drop missing values
    print('# outliers:', len(outliers_indices))    
    percent_outliers = 100*len(outliers_indices) / float(len(col.dropna()))
    print('% outliers:', percent_outliers)
    
    return outliers_indices, len(outliers_indices), percent_outliers

