import numpy as np
import pandas as pd


## Check collinearity over independent features.

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