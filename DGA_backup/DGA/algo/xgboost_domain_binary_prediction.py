import threading
print('Acitve Threads Working')
print(threading.activeCount())

import sys 
import json 
import numpy as np
import pandas as pd 
from pandas.tseries.offsets import *
import datetime
from datetime import datetime
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

warnings.filterwarnings("ignore")
start_d= datetime.now()
print('Start Time of Script: ')
start = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
print(r"""
            %s
            
        """  % (start))

print('features saved to file')

print('prediction now')

#DGA_good_bad_ugly_test = pd.read_csv('../data/features.csv')
DGA_good_bad_ugly_test = pd.read_csv('../data/DGA_good_bad_ugly_test_features_wordsegment.csv')
#DGA_good_bad_ugly_test = DGA_good_bad_ugly_test.sample(frac=.6).reset_index(drop=True)
DGA_good_bad_ugly_test = DGA_good_bad_ugly_test.drop_duplicates(['domain'], keep='last')
print('length of training data before split')
print(len(DGA_good_bad_ugly_test))


#take a sample of 75% of the full data
DGA_good_bad_ugly_test = DGA_good_bad_ugly_test.sample(frac=0.75, replace=True)
print('length of DGA_good_bad_ugly_test after sample: ')
print(len(DGA_good_bad_ugly_test))


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
import sys

#one hot for tld
onehot = pd.get_dummies(DGA_good_bad_ugly_test, columns=['tld',], prefix=['tld'])
X_predict2 = onehot.filter(['entropy', 'hyphen_count', 'string_len_query1', 'Vowels', 'Consonents', 
                           'syllables', 'digit_count', 'consec_vowel_ratio', 'dot_count', 'longest_conc_count'
                           'word_digit_ratio', 'tld_len', 'wordsegment_host_count','unique_char_count'])

X = X_predict2
y = onehot.filter(['good_bad'])
print('data transformation complete')

#train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

print("Shape of X_predict")
print(X_predict2.shape)

#XGBoost model 
from numpy import loadtxt
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#fit model
model4 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_wieght=1,
    seed=27)

kfold= StratifiedKFold(n_splits=3, random_state=123)
model4.fit(X_train, y_train)
results = cross_val_score(model4, X, y, cv=kfold)
print('Kfolds results: ')
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#make predictions for test data
y_pred = model4.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions 
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

import statsmodels.api as sm
X2 = X
y2 = y
X2 = X2.replace([np.inf, -np.inf], 0)
y2 = y2.replace([np.inf, -np.inf], 0)
est = sm.OLS(y2, X2)
est2 = est.fit()
print(est2.summary())

y_pred2 = list(y_pred)
cfm = confusion_matrix(y_test, y_pred2)
print(cfm)

print('#TRUE NEGATIVE  | FALSE POSITIVE')
print('#FALSE NEGATIVE | TRUE POSITIVE')

#Scoring the model
print('Accuracy score: ')
print(accuracy)
print(classification_report(y_test, predictions))
print('Precision - [TP/TP+FP]')
print('Recall  - [TP/TP+FN]')
print('f1-score  - [2*(Recall * Precision) / (Recall + Precision)')
print('support - number of samples of the true response that line in the class')

#feature importance 
print('feature importance: ')
feature_import = pd.DataFrame(data=model4.feature_importances_, index=X_train.columns.values, columns=['values'])
feature_import.sort_values(['values'], ascending=False, inplace=True)
print(feature_import.transpose())

f = 'gain'
gain = model4.get_booster().get_score(importance_type = f)
print('gain of features')
print(gain)

w = 'weight'
weight = model4.get_booster().get_score(importance_type = w)
print('weight of features')
print(gain)

#save model 
import pickle 
filename = ('xgboost_domain_binary.sav')
pickle.dump(model4, open(filename, 'wb'))
print('binary algo of DGA domains done')

import datetime
from datetime import datetime
ENDTIME_D = datetime.now()
print('ENDTIME')
print(str(ENDTIME_D))

print('RUN TIME')
print(ENDTIME_D - start_d)

