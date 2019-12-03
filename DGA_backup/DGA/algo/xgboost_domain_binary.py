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

#data input 
alexa_10mil = pd.read_csv('../data/top10milliondomains.csv')
alexa_10mil = pd.DataFrame(alexa_10mil)
alexa_10mil.columns = ['rank', 'domain', 'open_page_rank']
alexa_10mil_domain = alexa_10mil['domain']
alexa_10mil_domain.columns = ['domain']
alexa_10mil_domain = pd.DataFrame(alexa_10mil_domain)
print('Alexa top 5: ')
print(alexa_10mil.head())
#pull in alexa top 10 million 

good_urls = pd.DataFrame(alexa_10mil_domain)
#select top 1m
good_urls_top1m = good_urls.head(n=1000000)
#split after 1m
good_urls_after_top1m = good_urls.iloc[1000000:]
#sample after top 1m
good_urls_mix = good_urls_after_top1m.sample(frac=0.35, replace=True)
#append the good_urls_mix and good_urls_top1m
good_urls = pd.concat([good_urls_after_top1m, good_urls_mix])
#sleect 3mil good urls
good_urls = good_urls.sample(3000000)
#good urls get good or 0 label
good_urls['good_bad'] = 0 
good_urls.columns = ['resource', 'good_bad']
print('shape of good urls')
print(good_urls.shape)

#merge the good and bad + move bad_good to last column 
df = pd.read_csv('../data/domain_dga.csv')
df.columns = ['family','domain','time','source']

bad = df 
bad.rename(columns={'domain':'resource'}, inplace=True)
bad['good_bad'] = 1
bad = bad[['resource','good_bad']]

good_bad_ugly = good_urls.append(bad,ignore_index=True)

#move dependent variable good_bad to end
cols = list(good_bad_ugly.columns.values)
cols.pop(cols.index('good_bad'))
good_bad_ugly = good_bad_ugly[cols+['good_bad']]
#drop nas
good_bad_ugly = good_bad_ugly.dropna(axis=0, subset=['resource'])
print('length of good_bad_ugly')
print(len(good_bad_ugly))

#feature building
DGA_good_bad_ugly_test = good_bad_ugly
DGA_good_bad_ugly_test['resource'] = DGA_good_bad_ugly_test

#entropy
def entropy(string):
    #get prob of chars in string
    prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
    
    #calculate the entropy 
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    
    return entropy
DGA_good_bad_ugly_test['entropy'] = DGA_good_bad_ugly_test['resource'].apply(entropy)

#ip address
ippattern = (r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
DGA_good_bad_ugly_test['IP'] = DGA_good_bad_ugly_test['resource'].str.contains(ippattern)

#tld pattern
tldpattern = r'\.([^.\n\s]\w+)$'
DGA_good_bad_ugly_test['tld'] = DGA_good_bad_ugly_test['resource'].str.findall(tldpattern).astype(str)
DGA_good_bad_ugly_test['tld'] = DGA_good_bad_ugly_test['tld'].replace('[\[|\]|\/|\.|\']', '', regex=True, inplace=False)
DGA_good_bad_ugly_test['tld'] = DGA_good_bad_ugly_test['tld'].str.split(',').str[0]

#hyphen count
DGA_good_bad_ugly_test['hyphen_count'] = DGA_good_bad_ugly_test.resource.str.count('-')
#dot count
DGA_good_bad_ugly_test['dot_count'] = DGA_good_bad_ugly_test.resource.str.count(r'\.')
#string length of query1 
DGA_good_bad_ugly_test['string_len_query1'] = DGA_good_bad_ugly_test.resource.str.len()
#tld length
DGA_good_bad_ugly_test['tld_len'] = DGA_good_bad_ugly_test['tld'].str.len()

#count of vowels and consonents 
vowels = set("aeiou")
cons = set("bcdfghjklmnpqrstvwxyz")
DGA_good_bad_ugly_test['Vowels'] = [sum(1 for c in x if c in vowels) for x in DGA_good_bad_ugly_test['resource']]
DGA_good_bad_ugly_test['Consonents'] = [sum(1 for c in x if c in cons) for x in DGA_good_bad_ugly_test['resource']]

#count the number of syllables in a word 
import re 
def syllables(word):
    word = word.lower()
    if word.endswith('e'):
        word = word[:-1]
    count = len(re.findall('[aeiou]+', word))
    return count

DGA_good_bad_ugly_test['syllables'] = DGA_good_bad_ugly_test['resource'].apply(syllables)

#count the number of digets in domain 
DGA_good_bad_ugly_test['digit_count'] = DGA_good_bad_ugly_test['resource'].apply(lambda x: len([s for s in x if s.isdigit()]))

#extract host, domain, subdomains 
DGA_good_bad_ugly_test['subdomain'] = DGA_good_bad_ugly_test['resource'].apply(lambda url: tldextract.extract(url).subdomain)
DGA_good_bad_ugly_test['domain'] = DGA_good_bad_ugly_test['resource'].apply(lambda url: tldextract.extract(url).domain)
DGA_good_bad_ugly_test['host'] = DGA_good_bad_ugly_test[['subdomain', 'domain']].apply(lambda x: '.'.join(x), axis=1)
DGA_good_bad_ugly_test['host'] = DGA_good_bad_ugly_test['host'].str.replace(r'^\.', '').astype(str)

#consonents to vowels ratio
DGA_good_bad_ugly_test['consec_vowel_ratio'] = (DGA_good_bad_ugly_test['Vowels'] / DGA_good_bad_ugly_test['Consonents']).round(5)

#count the longest string without vowels 
from itertools import groupby
is_vowel = lambda char: char in r'aeious0123456789\:\.\-\\'

def suiteConsonnes(in_str):
    return ["".join(g) for v, g in groupby(in_str, key=is_vowel) if not v]

string_len = DGA_good_bad_ugly_test['resource'].apply(suiteConsonnes)
string_len = pd.DataFrame(string_len).reset_index()
string_len.columns = ['index', 'consec_cons']

def longestconc(word):
    for letter in word:
        return max(word, key=len)
    
string_len['longest_conc'] = string_len['consec_cons'].apply(longestconc)
string_len['longest_conc_count'] = string_len['longest_conc'].str.len()
string_len['longest_conc_count'] = string_len['longest_conc_count'].fillna(0)
string_len = pd.DataFrame(string_len)

DGA_good_bad_ugly_test['consec_vowel_ratio'] = DGA_good_bad_ugly_test['consec_vowel_ratio'].fillna(0)

#move dependent variable (good_bad) to end
cols = list(DGA_good_bad_ugly_test.columns.values)
cols.pop(cols.index('good_bad'))
DGA_good_bad_ugly_test = DGA_good_bad_ugly_test[cols+['good_bad']]

DGA_good_bad_ugly_test = DGA_good_bad_ugly_test.reset_index()
DGA_good_bad_ugly_test = DGA_good_bad_ugly_test.merge(string_len, on='index')

#replace long_conc_count if over 10
#DGA_good_bad_ugly_test['long_conc_count'] = DGA_good_bad_ugly_test['long_conc_count'].fillna(0)

#count digits and alpha (letters) in domain
def count_digits(string):
    return sum(item.isdigit() for item in string)

def count_words(string):
    return sum(item.isalpha() for item in string)

DGA_good_bad_ugly_test['alpha_count'] = DGA_good_bad_ugly_test['resource'].apply(count_words)



print('begning wordsegment')
wordsegment_start = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
print('wordsegment start')
print(wordsegment_start)
import wordsegment
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

print(DGA_good_bad_ugly_test.head())

#write features to csv for saving
DGA_good_bad_ugly_test.to_csv('/home/ubuntu/python/DGA/data/features.csv')

print('features saved to file')
print('prediction now')

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
filename = ('./xgboost_domain_binary.sav')
pickle.dump(model4, open(filename, 'wb'))
print('binary algo of DGA domains done')

import datetime
from datetime import datetime
ENDTIME_D = datetime.now()
print('ENDTIME')
print(str(ENDTIME_D))

print('RUN TIME')
print(ENDTIME_D - start_d)
