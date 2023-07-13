#!/account/tli/anaconda3/bin/python


import time
start_time = time.time()

import sys
var="2"
seeds="0"

###Loading packages
import warnings
warnings.filterwarnings('ignore')



# Set a seed value
seed_value= int(seeds)#41
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)


from keras import backend as K
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


import itertools
import pandas as pd


from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras import initializers
from keras.regularizers import l1, l2
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

he_normal = initializers.he_normal(seed=int(seed_value))

def monitor_f(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


# define base model
def create_model(n_dim, node, activation, optimizer):

    # create model
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(node, input_dim = n_dim, kernel_initializer=he_normal, activation=activation, kernel_regularizer=l2(0.01)))#, kernel_regularizer=l2(0.001)
    NN_model.add(BatchNormalization())
    NN_model.add(Dropout(0.2))

    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer=he_normal, activation='sigmoid'))

    # Compile model
    NN_model.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])#, monitor_f
    return NN_model

###fit model
def fit_model(X_train, y_train, X_validation, y_validation, n, model_path, model, batch_size):
    ###balanced class weight
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train),y_train)
    class_weights = {0:1,1:6}
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)#monitor_f
    ###define checkpoint for the best model
    checkpoint = ModelCheckpoint(model_path, verbose=1, monitor='loss',save_best_only=True, mode='min')#val_acc,monitor_f
    ###fit model
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=n, batch_size=batch_size, class_weight=class_weights, callbacks=[checkpoint, es], shuffle=False)#, es 
    ###load the best model
    best_model = load_model(model_path)#, custom_objects={'monitor_f': monitor_f}
    return best_model#model

def model_predict(X, y, model, col_name):
    y_pred = model.predict(X)
    y_pred_class = np.where(y_pred > 0.65, 1, 0)
    pred_result = pd.DataFrame()
    pred_result['id'] = y.index
    pred_result['y_true'] = y.values
    pred_result['prob_'+col_name] = y_pred
    pred_result['class_'+col_name] = y_pred_class

    result=measurements(y, y_pred_class, y_pred)
    return pred_result, result

def measurements(y_test, y_pred, y_pred_prob):
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob) 
    sensitivity = metrics.recall_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    specificity = TN/(TN+FP)
    npv = TN/(TN+FN)
    return [TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc, balanced_accuracy]
    

def dim_reduce(df, test_df, model_path, col_name1):
    
    X_org = df.iloc[:, 3:]
    #print(X_org.shape)
    y_org = df.loc[:, 'y_true']
    X_test = test_df.iloc[:, 3:]
    #print(X_test.shape)
    y_test = test_df.loc[:, 'y_true']

    X, X_val, y, y_val = train_test_split(X_org,  y_org, test_size=0.2, stratify=y_org, random_state=int(var))
    #print("X train shape: ", X.shape)

    sc = StandardScaler()
    #sc = MinMaxScaler()
    sc.fit(X)
    X = sc.transform(X)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)    


    ### load model
    #model_path = '/account/tli/ames/result/deepames_sensitivity/result_fixed/weights/seed_0_neuron_32_lr_0.1_bs_32_activation_relu_epoch_100_weight0_0.5_para_2_weight_16_weights.h5'#for deepAmes plus
    best_model = load_model(model_path)
    
    ### predict
    validation_class, validation_result = model_predict(X_val, y_val, best_model, col_name1)
    test_class, test_result = model_predict(X_test, y_test, best_model, col_name1)

    K.clear_session()
    tf.reset_default_graph() 

    return validation_class, validation_result, test_class, test_result

def sep_performance(df):
    cols = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'MCC', 'BA']
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

def reform_result(results):
    df = pd.DataFrame(data=results.items())
    ###reform the result data format into single colum
    df = df.rename(columns={0:'name', 1:'value'})
    df['name'] = df['name'].astype('str')
    df['value'] = df['value'].astype('str')
    df = sep_performance(df)
    return df


def deepames_prediction(probability_path, name, weight, result_path):
    ###data
    data = pd.read_csv(probability_path+'/validation_probabilities_' + name + '.csv')
    test = pd.read_csv(probability_path+'/test_probabilities_' + name + '.csv')
    #print('data: ' ,data.shape)
    #print('test: ' ,test.shape)


    ### set path
    path2 = result_path + '/validation_class'
    path3 = result_path + '/validation_performance'
    path4 = result_path + '/test_class'
    path5 = result_path + '/test_performance'

    #initial performance dictionary
    validation_results={}
    test_results={}

    col_name = 'weight'+str(weight)
    model_path = '/account/tli/ames/scripts/hpc/a_outside_validation_deepames/deepamesPlus_weights' + '/weight_'+str(weight)+'.h5'

    ###get the prediction
    validation_class, validation_result, test_class, test_result = dim_reduce(data, test, model_path, col_name)


    validation_results[col_name]=validation_result
    validation_class.to_csv(path2+'/validation_'+col_name+'.csv')

    test_results[col_name]=test_result
    test_class.to_csv(path4+'/test_'+col_name+'.csv')

    reform_result(validation_results).to_csv(path3+'/dnn_validation_para_'+ var + '_' + col_name+'.csv')
    reform_result(test_results).to_csv(path5+'/dnn_test_para_'+ var + '_' + col_name+'.csv')
    

print("--- %s seconds ---" % (time.time() - start_time))
print("DeepAmes prediction is completed")
