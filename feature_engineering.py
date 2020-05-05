#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from datetime import datetime, timedelta
from pytz import timezone


## Transform a feature via one-hot encoding.

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