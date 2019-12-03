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

DGA_good_bad_ugly_test = pd.read_csv('/home/ubuntu/python/DGA/data/DGA_good_bad_ugly_test_features.csv')

print('begning wordsegment')
wordsegment_start = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
print('wordsegment start')
print(wordsegment_start)
from wordsegment import load, segment, clean
load()
DGA_good_bad_ugly_test['wordsegment_host'] = DGA_good_bad_ugly_test.host.apply(segment)
#count the elements in wordsegment_host list 
DGA_good_bad_ugly_test['wordsegment_host_count'] = DGA_good_bad_ugly_test['wordsegment_host'].map(len)
wordsegment_finish = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
print('*wordsegment finish*:')
print(wordsegment_finish)

#character frequency in a domain
def char_frequency(str1):
    dict = {}
    for n in str1:
        keys = dict.keys()
        if n in keys:
            dict[n] += 1
        else:
            dict[n] = 1
    return dict.keys()

DGA_good_bad_ugly_test['unique_char_count'] = DGA_good_bad_ugly_test['host'].apply(char_frequency).map(len)


#move dependent variable (good_bad) to end
cols = list(DGA_good_bad_ugly_test.columns.values)
cols.pop(cols.index('good_bad'))
DGA_good_bad_ugly_test = DGA_good_bad_ugly_test[cols+['good_bad']]

#cleanup of features and normalization

#to add to feature engineering above
DGA_good_bad_ugly_test['word_digit_ratio'] = (DGA_good_bad_ugly_test['digit_count'] / DGA_good_bad_ugly_test['alpha_count']).round(5)


#replace info for features
DGA_good_bad_ugly_test.consec_vowel_ratio = DGA_good_bad_ugly_test.consec_vowel_ratio.replace([np.inf, -np.inf], 0)
DGA_good_bad_ugly_test.entropy = DGA_good_bad_ugly_test.entropy.replace([np.inf, -np.inf], 0)
DGA_good_bad_ugly_test.longest_conc_count = DGA_good_bad_ugly_test.longest_conc_count.fillna(0)
DGA_good_bad_ugly_test.consec_vowel_ratio = DGA_good_bad_ugly_test.consec_vowel_ratio.fillna(0)
DGA_good_bad_ugly_test.word_digit_ratio = DGA_good_bad_ugly_test.word_digit_ratio.fillna(0)
DGA_good_bad_ugly_test.word_digit_ratio = DGA_good_bad_ugly_test.word_digit_ratio.replace([np.inf, -np.inf], 0)

print('sending features with wordsegment to csv')

DGA_good_bad_ugly_test.to_csv('/home/ubuntu/python/DGA/data/DGA_good_bad_ugly_test_features_wordsegment.csv')
