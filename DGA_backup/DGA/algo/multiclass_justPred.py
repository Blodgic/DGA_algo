#!/usr/bin/env python3


#prediction
import pandas_profiling
import pandas as pd
import numpy as np 
import json
from pandas.tseries.offsets import *
from datetime import date, timedelta
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import statsmodels.discrete.discrete_model as sm
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import os
import re
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import warnings
import math
import tldextract
import logging
import datetime
from datetime import datetime

#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 2000)


#warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s',filename='multiclass_pred_dga.log', filemode='w')
logging.info("MultiClass Logging Begins")
start_d= datetime.now()
start = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
print(r"""
            %s
            
        """  % (start))

DGA_good_bad_ugly_test = pd.read_csv('../data/DGA_good_bad_ugly_test_features_wordsegment.csv')
#DGA_good_bad_ugly_test = DGA_good_bad_ugly_test.sample(frac=.6).reset_index(drop=True)
DGA_good_bad_ugly_test = DGA_good_bad_ugly_test.drop_duplicates(['domain'], keep='last')
print('length of training data before split')
print(len(DGA_good_bad_ugly_test))


print('training begins')
#one hot for tld
onehot = pd.get_dummies(DGA_good_bad_ugly_test, columns=['tld',], prefix=['tld'])
X_predict8 = onehot.filter(['entropy', 'hyphen_count', 'string_len_query1', 'Vowels', 'Consonents', 
                           'syllables', 'digit_count', 'consec_vowel_ratio', 'dot_count', 'longest_conc_count'
                           'word_digit_ratio', 'tld_len', 'wordsegment_host_count','unique_char_count'])

X = X_predict8
y = onehot.filter(['family'])
print('data transformation complete')

#train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

print("Shape of X_predict8")
print(X_predict8.shape)

#determine the number of classes 
print('number of samples and features')
print(len(DGA_good_bad_ugly_test.family.value_counts()))
print('number of unique families')
class_count = len(DGA_good_bad_ugly_test.family.value_counts())

print('original X_train, and y_train shape')
X_train.shape, y_train.shape

print('y_train family counts')
print(y_train.family.value_counts())

#XGBoost model 
from numpy import loadtxt
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#fit model
classifier = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softmax',
    num_class=class_count,
    nthread=4,
    scale_pos_wieght=1,
    seed=123)

#kfold= StratifiedKFold(n_splits=3, random_state=123)
classifier.fit(X_train, y_train)

#save model 
import pickle 
filename = ('multiclass_xgboost.sav')
pickle.dump(classifier, open(filename, 'wb'))
print('multiclass algo of DGA domains done')

print('**ON TO ACCURACY*')
y_pred=classifier.predict(X_test)
print('accuracy: ')
accuracy = accuracy_score(y_test, y_pred)

print('made it to results')
print('Classification Report')
print(classification_report(y_test, y_pred))
print('Precision - [TP/TP+FP]')
print('Recall  - [TP/TP+FN]')
print('f1-score  - [2*(Recall * Precision) / (Recall + Precision)')
print('support - number of samples of the true response that line in the class')

import datetime
from datetime import datetime

print('END TIME')
ENDTIME_d = datetime.now()
ENDTIME = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
print(ENDTIME)

print('TOTAL RUN TIME')
diff = ENDTIME_d - start_d
print('****************')
print(diff)

