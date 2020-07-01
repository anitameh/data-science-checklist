#!/usr/bin/env python3

import numpy as np
import pandas as pd

import scipy.stats as ss
import scipy.special as sp

import matplotlib.pyplot as plt
import seaborn as sns


## Basic initial stats.

def get_alldata_stats(df, datecol=""):
    """Get summary statistics about the all data.
    Args:
        df (pandas.DataFrame): Entire dataset.
        datecol (numpy.ndarray): Column of dataset.
    """
    # number of rows in dataframe
    num_rows = len(df)
    print('total # of rows: {} \n'.format(num_rows))

    # number of columns in entire dataframe
    num_cols = len(df.columns)
    print('total # of columns: {} \n'.format(num_cols))
    print(list(df.columns))

    # date range for data
    if datecol:
        min_date = min(df[datecol])
        max_date = max(df[datecol])
        print(min_date)
        print(max_date)

    # coverage
    total_percent_missing = (100*df.isna().sum().sum() / float(len(df)*len(df.columns)))
    print("\n{:,.2f}% of the data is missing \n".format(round(total_percent_missing, 2)))

    # fully-empty columns
    print(df.isna().sum())

    return (num_rows, num_cols, total_percent_missing)


def get_onecol_stats(df, col):
    """Get summary statistics about the specified data column.
    Args:
        df (pandas.DataFrame): Entire dataset.
        col (numpy.ndarray): Column of dataset.
    """

    # number of unique items
    num_unique = len(np.unique(df[col]))
    print(num_unique)
    
    # mean, std without NaN values
    mean = np.nanmean(df[col])
    std = np.nanstd(df[col])
    print('mean of {} = {}'.format(col, mean))
    print('std of {} = {}'.format(col, std))

    return (num_unique, mean, std)


def report_column_coverage(df, col):
    """Report missing values.

    Args:
        df (pandas.DataFrame): Entire dataset.
        col (numpy.ndarray): Column of dataset.
        
    """
    count_missing_vals = df[col].isna().sum()
    print('# missing values:', count_missing_vals)
    
    percent_missing = 100*count_missing_vals / float(len(df[col]))
    print('% missing values:', percent_missing)
    
    print('# values:', len(col))
    
    return count_missing_vals, percent_missing, len(col)


## Determining outliers.

# Boxplot method. Why? Z-score biased. Other methods? DBScan, Isolation forests.
def plot_outliers(df, col, title='', xlabel=''):
    """Visualizes outliers with a boxplot.
    
    Args:
        df (pandas.DataFrame): Entire dataset.
        col (numpy.ndarray): Column of dataset.
        title (String): Title to use for plot.
        xlabel (String): Label to use for x-axis.
    
    Returns:
        Boxplot of data in col.
    """
    plt.figure(figsize=[16,4])
    sns.boxplot(df[col])
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
    
    nonoutliers_indices = np.where(col.dropna() <= upper_bound)[0]
    
    return outliers_indices, nonoutliers_indices, len(outliers_indices), percent_outliers


