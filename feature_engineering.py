#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from datetime import datetime, timedelta
from pytz import timezone


## Transform a feature via one-hot encoding.
## use one-hot encoding over label encoding if order doesn't matter
## (https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/)

def one_hot_encode(data, col):
    """Converts inputted col into a data frame reflecting one-hot encoding.
    
    Args:
        data (pandas.DataFrame): Dataset.
        col (numpy.ndarray): Column of dataset.
    
    Returns:
        Data frame with one hot encoding
    """
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(np.array(data[col]).reshape(-1, 1))

    column_names = []
    for each_cat in encoder.categories_[0]:  # this is a list with an array as the first elt
        column_names.append(col + '_' + str(each_cat))

    ohe_col = pd.DataFrame(encoder.transform(np.array(data[col]).reshape(-1, 1)).toarray(), columns=column_names)    
    display(ohe_col.head(10))

    return ohe_col


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


def convert_string_to_datetime(s, timezone_str='America/New_York'):
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


def convert_datetime_to_number(d, timezone_str='America/New_York'):
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