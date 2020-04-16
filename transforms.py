import numpy as np
import pandas as pd

import scipy.stats as ss
import scipy.special as sp
from sklearn.preprocessing import OneHotEncoder

from datetime import datetime, timedelta
from pytz import timezone


## Handling timestamps.
def convert_string_to_hour(s):
    '''
    :input string / float s: timestamp / NaN
    :returns numeric: hour for string of timestamp; otherwise NaN
    '''
    if isinstance(s, float):
        return s
    else:
        return int(s[11:13])


def convert_string_to_datetime(s, timezone_str):
    '''
    :input string / float s: timestamp / NaN
    :input string timezone: e.g. 'US/Pacific'
    :returns datetime:
    '''
    if isinstance(s, float):
        return s
    else:
        datetime_object = datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S')
        return datetime_object.replace(tzinfo=timezone(timezone_str))


def convert_datetime_to_number(d, timezone_str):
    '''
    Ideally input should be a datetime object. 
    if not, return it. that's likely because it's a nan.
    :input datetime obj d: includes timezone
    :input string timezone: e.g. 'US/Pacific'
    :returns timedelta in seconds:
    '''
    if isinstance(d, float):
        return d
    else:
        special_date = datetime(1970, 1, 1, tzinfo=timezone(timezone_str))
        return (d - special_date).total_seconds()
    
def convert_timestamp_to_date(ts):
    '''
    Check timezone.
    :input int ts: timestamp
    '''
    if isinstance(ts, float):
        return ts
    else:
        return datetime.fromtimestamp(ex_d_int)


## Checking coverage.

def report_missingness(col):
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


## Checking collinearity.

def derive_collinearity_matrix(independent_features):
    '''
    :input list of strings independent_features:
    '''
    matrix = []
    for each_featureA in independent_features:
        row = []
        for each_featureB in independent_features:
            corr, _ = ss.pearsonr(subset[each_featureA], subset[each_featureB])
            row.append(corr)
        matrix.append(row)

    return pd.DataFrame(matrix, columns=independent_features)


## Transforming a feature via one-hot encoding.

def one_hot_encode(data, col):
    '''
    :input pandas.DataFrame data:
    :input numpy.ndarray col:
    '''
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(np.array(data[col]).reshape(-1, 1))
    print(encoder.categories_)

    column_names = []
    for each_cat in encoder.categories_:
        column_names.append(col + '_' + str(each_cat))

    ohe_cancel_penalty = pd.DataFrame(encoder.transform(np.array(data[col]).reshape(-1, 1)).toarray(), columns=column_names)    
    display(ohe_cancel_penalty.head())

    return ohe_cancel_penalty
