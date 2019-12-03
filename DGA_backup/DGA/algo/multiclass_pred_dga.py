#!/usr/bin/env python3
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
import logging


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
#data input 
alexa_10mil = pd.read_csv('/home/ubuntu/python/DGA/data/top10milliondomains.csv')
alexa_10mil = pd.DataFrame(alexa_10mil)
alexa_10mil.columns = ['rank', 'domain', 'open_page_rank']
alexa_10mil_domain = alexa_10mil['domain']
alexa_10mil_domain.columns = ['domain']
alexa_10mil_domain = pd.DataFrame(alexa_10mil_domain)

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
df = pd.read_csv('/home/ubuntu/python/DGA/data/domain_dga.csv')
df.columns = ['family','domain','time','source']

bad = df 
bad.rename(columns={'domain':'resource'}, inplace=True)
bad['good_bad'] = 1
bad = bad[['resource','family','good_bad']]
print('length of DGA domains')
print(len(bad))

#scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

family_count = bad.family.value_counts()
family_count = pd.DataFrame(family_count)
family_count = family_count.reset_index()
family_count.columns = ['family', 'count']
family_count.count = family_count['count'].astype(int)
print('**family count**')
class_count = len(family_count)
print(len(family_count))
family_count_1000 = family_count[family_count.count > 999]
print('length of families over 999')
print(len(family_count_1000))
print('original length of bad')
print(len(bad))
bad_family = (pd.merge(bad,family_count_1000, on='family'))
print('DGA family counts')
print(bad_family.family.value_counts())
print('new length of bad_family')

#good domains have family name of benign
good_urls['family'] = 'benign'

good_bad_ugly = good_urls.append(bad_family,ignore_index=True)

#move dependent variable good_bad to end
cols = list(good_bad_ugly.columns.values)
cols.pop(cols.index('good_bad'))
good_bad_ugly = good_bad_ugly[cols+['good_bad']]

#drop nas
good_bad_ugly = good_bad_ugly.dropna(axis=0, subset=['resource'])
print('length of good_bad_ugly')
print(len(good_bad_ugly))

pd.options.display.float_format = '{:20,.4f}'.format
warnings.filterwarnings("ignore")

print("good bad ugly top 5")

#feature building
DGA_good_bad_ugly_test = good_bad_ugly
#convert to string
DGA_good_bad_ugly_test['resource'] = DGA_good_bad_ugly_test['resource'].astype(str)

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
filename = ('/home/ubuntu/python/DGA/algo/multiclass_xgboost.sav')
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

