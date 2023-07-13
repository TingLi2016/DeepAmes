#!/account/tli/anaconda3/bin/python

import time
start_time = time.time()

### Warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

### import scripts
import base_knn
import base_lr
import base_svm
import base_rf
import base_xgboost

import select_base

import validation_predictions_combine
import test_predictions_combine

import deepames_plus

### define the path for data, base classifers, dnn results 
features = pd.read_csv('/account/tli/ames/data/mold2_selected_features_10444.csv').feature.unique()# path for mold2_selected_features_10523.csv
data = pd.read_csv('/account/tli/ames/data/clean_train_10444.csv', low_memory=False)# path for clean_train_10444.csv 
external =  pd.read_csv('/account/tli/ames/scripts/hpc/a_outside_validation_deepames/data/external_mold2.txt', sep='\t') # path for ames_test.csv for test set
external['label'] = np.where(external.index < 3000, 0, 1)

name = 'ames6512' # can be any name 

workDir = '/account/tli/ames/scripts/hpc/a_outside_validation_deepames/result/external6512'
base_path = workDir + '/base'# path for base classifiers
probability_path = workDir + '/probabilities_output' # path for the combined probabilities (model-level representations)
result_path = workDir + '/result' # path for the final deepcarc predictions

### run the scripts
base_knn.generate_baseClassifiers(features, data, external, base_path+'/knn')
base_lr.generate_baseClassifiers(features, data, external, base_path+'/lr')
base_svm.generate_baseClassifiers(features, data, external, base_path+'/svm')
base_rf.generate_baseClassifiers(features, data, external, base_path+'/rf')
base_xgboost.generate_baseClassifiers(features, data, external, base_path+'/xgboost')

mcc = select_base.select_base_classifiers(base_path)

validation_predictions_combine.combine_validation_probabilities(base_path, mcc, probability_path, name)
test_predictions_combine.combine_test_probabilities(base_path, mcc, probability_path, name)

for weight in range(6, 19, 1):
    deepames_plus.deepames_prediction(probability_path, name, weight, result_path)



print("--- %s seconds ---" % (time.time() - start_time))
