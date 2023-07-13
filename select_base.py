import pandas as pd
import numpy as np

import os
from os import listdir
from os.path import isfile, join
import itertools
from functools import reduce
import os.path

def sep_performance(filepath):
    df = pd.read_csv(filepath)
    df = df.iloc[:, 1:]
    df = df.rename(columns={'0':'name', '1':'value'})
    df['name'] = df['name'].astype('str')
    df['value'] = df['value'].astype('str')
    df['model'] = df['name'].str.split('_').str[0].values
    df['seed'] = df['name'].str.split('_paras_').str[0].str.split('_seed_').str[1]
    
    
    cols = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'MCC']
    for i, col in enumerate(cols):
        if i == 0:
            df[col] = df.value.str.split(',').str[i].str.split('[').str[1].values
        elif i == len(cols)-1:
            df[col] = df.value.str.split(',').str[i].str.split(']').str[0].values
        else:
            df[col] = df.value.str.split(',').str[i].values

    for i, col in enumerate(cols):
        if i < 4:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
            df[col] = round(df[col], 3)
    del df['value']
            
    return df

def select_base_classifiers(basepath):

    xgboost = sep_performance(basepath + '/xgboost/validation_performance/validation_xgboost_paras_151.csv')
    rf = sep_performance(basepath + '/rf/validation_performance/validation_rf__paras_148.csv') 
    svm = sep_performance(basepath + '/svm/validation_performance/validation_svm_paras_19.csv') 
    lr = sep_performance(basepath + '/lr/validation_performance/validation_lr_paras_22.csv')
    knn = sep_performance(basepath + '/knn/validation_performance/validation_knn_paras_0.csv')

    result = pd.concat([xgboost, rf, svm, lr, knn], axis=0)
    
    subresult = result[(result.MCC <np.percentile(result.MCC.values, 95)) & (result.MCC > np.percentile(result.MCC.values, 5))]
    subresult = subresult.reset_index(drop=True)
    
    subresult.to_csv(basepath + '/selected_base_classifiers.csv')
    return subresult