import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


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