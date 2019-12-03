from flask import Flask, request, render_template, redirect, flash, url_for, jsonify
import flask
import logging
import sys 
import json 
import numpy as np
import pandas as pd 
from pandas.tseries.offsets import *
import datetime
from email.mime.text import MIMEText
from base64 import urlsafe_b64encode
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
import google.oauth2.credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from .config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from flask_mail import Mail, Message
from flask_migrate import Migrate
from flask_admin import Admin, AdminIndexView, helpers, expose
from flask_admin.menu import MenuLink

pd.options.display.float_format = '{:20,.4f}'.format
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config.from_object(Config)
login_manager = LoginManager(app)
db = SQLAlchemy(app)
mail = Mail(app)
migrate = Migrate(app, db)
from .models import *
from .forms import *
from .admin import *
from .tokenauth import token_auth, tokens as authtoken
from .email_api import CLIENT_SECRETS_FILE, SCOPES


@app.route('/')
def index():
    if current_user.is_authenticated:
        user_id = current_user.id
    else:
        user_id = None
    record = SiteRequest(user_id=user_id, ip_address=request.remote_addr, page=request.path)
    db.session.add(record)
    db.session.commit()
    return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        if current_user.is_authenticated:
            record = APIRequest(user_id=current_user.id, ip_address=request.remote_addr, query=request.form['domain'])
        else:
            record = APIRequest(user_id=None, ip_address=request.remote_addr, query=request.form['domain'])
        db.session.add(record)
        db.session.commit()
        result = request.form
        df = pd.DataFrame.from_dict(result, orient='index')
        #transform weird host entries
        df[0] = df[0].str.lower()
        df[0] = df[0].str.replace(r'(^hxxps?\:\/\/)', '')
        df[0] = df[0].str.replace(r'(^https?\:\/\/)', '')
        df[0] = df[0].str.replace(r'(^www)','')
        df[0] = df[0].str.replace(r'(^www\d+)','')
        df[0] = df[0].str.replace(r'([\[|\]])','')
        df[0] = df[0].str.split(r'\/').str[0]
        df[0] = df[0].str.replace(r'^\.','')

        #entropy
        def entropy(string):
                            #get prob of chars in string
                            prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]

                            #calculate the entropy
                            entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])

                            return entropy

        df['entropy'] = df[0].apply(entropy)

        #ip address
        ippattern = (r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
        df['IP'] = df[0].str.contains(ippattern)

        #tld pattern
        tldpattern = r'\.([^.\n\s]\w+)$'
        df['tld'] = df[0].str.findall(tldpattern).astype(str)
        df['tld'] = df['tld'].replace('[\[|\]|\/|\.|\']', '', regex=True, inplace=False)
        df['tld'] = df['tld'].str.split(',').str[0]

        #hyphen count
        df['hyphen_count'] = df[0].str.count('-')
        #dot count
        df['dot_count'] = df[0].str.count(r'\.')
        #string length of query1
        df['string_len_query1'] = df[0].str.len()
        #tld length
        df['tld_len'] = df['tld'].str.len()

        #count of vowels and consonents
        vowels = set("aeiou")
        cons = set("bcdfghjklmnpqrstvwxyz")
        df['Vowels'] = [sum(1 for c in x if c in vowels) for x in df[0]]
        df['Consonents'] = [sum(1 for c in x if c in cons) for x in df[0]]

        #count the number of syllables in a word
        import re
        def syllables(word):
            word = word.lower()
            if word.endswith('e'):
                word = word[:-1]
            count = len(re.findall('[aeiou]+', word))
            return count

        df['syllables'] = df[0].apply(syllables)

        #count the number of digets in domain
        df['digit_count'] = df[0].apply(lambda x: len([s for s in x if s.isdigit()]))

        #extract host, domain, subdomains
        df['subdomain'] = df[0].apply(lambda url: tldextract.extract(url).subdomain)
        df['domain'] = df[0].apply(lambda url: tldextract.extract(url).domain)
        df['host'] = df[['subdomain', 'domain']].apply(lambda x: '.'.join(x), axis=1)
        df['host'] = df['host'].str.replace(r'^\.', '').astype(str)

        #consonents to vowels ratio
        df['consec_vowel_ratio'] = (df['Vowels'] / df['Consonents']).round(5)

        #count the longest string without vowels
        from itertools import groupby
        is_vowel = lambda char: char in r'aeious0123456789\:\.\-\\'

        def suiteConsonnes(in_str):
            return ["".join(g) for v, g in groupby(in_str, key=is_vowel) if not v]

        string_len = df[0].apply(suiteConsonnes)
        string_len = pd.DataFrame(string_len).reset_index()
        string_len.columns = ['index', 'consec_cons']

        def longestconc(word):
            for letter in word:
                return max(word, key=len)

        string_len['longest_conc'] = string_len['consec_cons'].apply(longestconc)
        string_len['longest_conc_count'] = string_len['longest_conc'].str.len()
        string_len['longest_conc_count'] = string_len['longest_conc_count'].fillna(0)
        string_len = pd.DataFrame(string_len)

        df['consec_vowel_ratio'] = df['consec_vowel_ratio'].fillna(0)

        #move dependent variable (good_bad) to end
        #cols = list(df.columns.values)
        #cols.pop(cols.index('good_bad'))
        #df = df[cols+['good_bad']]

        df = df.reset_index()
        df = df.merge(string_len, on='index')

        #replace long_conc_count if over 10
        #df['long_conc_count'] = df['long_conc_count'].fillna(0)

        #count digits and alpha (letters) in domain
        def count_digits(string):
            return sum(item.isdigit() for item in string)

        def count_words(string):
            return sum(item.isalpha() for item in string)

        df['alpha_count'] = df[0].apply(count_words)
        df['word_digit_ratio'] = (df['digit_count'] / df['alpha_count']).round(5)
        #print('begning wordsegment')
        #wordsegment_start = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
        #print('wordsegment start')
        #print(wordsegment_start)
        from wordsegment import load, segment, clean
        load()
        df['wordsegment_host'] = df.host.apply(segment)
        #count the elements in wordsegment_host list
        df['wordsegment_host_count'] = df['wordsegment_host'].map(len)
        wordsegment_finish = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
        #print('*wordsegment finish*:')
        #print(wordsegment_finish)

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

        df['unique_char_count'] = df['host'].apply(char_frequency).map(len)
        print('domain tried: ')
        print(df.iloc[0])
        #normalize the data
        #replace info for features
        df.consec_vowel_ratio = df.consec_vowel_ratio.replace([np.inf, -np.inf], 0)
        df.entropy = df.entropy.replace([np.inf, -np.inf], 0)
        df.longest_conc_count = df.longest_conc_count.fillna(0)
        df.consec_vowel_ratio = df.consec_vowel_ratio.fillna(0)
        df.word_digit_ratio = df.word_digit_ratio.fillna(0)
        df.word_digit_ratio = df.word_digit_ratio.replace([np.inf, -np.inf], 0)

        #move dependent variable (good_bad) to end
        #cols = list(df.columns.values)
        #cols.pop(cols.index('good_bad'))
        #df = df[cols+['good_bad']]

        #add training features and split to training/test
        #one hot for tld
        testme = df
        onehot = pd.get_dummies(df, columns=['tld',], prefix=['tld'])
        X_predict2 = onehot.filter(['entropy', 'hyphen_count', 'string_len_query1', 'Vowels', 'Consonents',
                                   'syllables', 'digit_count', 'consec_vowel_ratio', 'dot_count', 'longest_conc_count'
                                   'word_digit_ratio', 'tld_len', 'wordsegment_host_count','unique_char_count'])

        #pickup and load model against features
        if app.config['ENV'] == 'development':
            filename = ('../algo/xgboost_domain_binary.sav')
            filename2 = '../algo/multiclass_xgboost.sav'
        else:
            filename = ('/home/ubuntu/python/DGA/algo/xgboost_domain_binary.sav')
            filename2 = '/home/ubuntu/python/DGA/algo/multiclass_xgboost.sav'
        loaded_model = joblib.load(filename)
        result = loaded_model.predict_proba(X_predict2)
        results = pd.DataFrame(result)
        results.columns = ['benign', 'DGA']
        algo = testme.join(results)
        algo_url = algo[['benign', 'DGA']].copy()
        df = df.join(algo_url)
        loaded_model2 = joblib.load(filename2)
        result_class = loaded_model2.predict(X_predict2)
        result_class = pd.DataFrame(result_class)
        result_class.columns = ['DGA_family']
        df = df.join(result_class)
        df.loc[(df['DGA'] >= 0.5) & (df['DGA_family'] == 'benign'), 'DGA_family'] = 'unknown'
        #import DGA intel csv
        if app.config['ENV'] == 'development':
            domain_dga = pd.read_csv('../data/domain_dga.csv')
        else:
            domain_dga = pd.read_csv('/home/ubuntu/python/DGA/data/domain_dga.csv')
        domain_dga['domain'] = domain_dga['domain'].astype(str)
        #the match function to bring back all rows
        answer = df[0].isin(domain_dga['domain'])
        answer = pd.DataFrame(answer)
        answer.columns = ['DGA_match']
        match = domain_dga.loc[domain_dga['domain'].isin(df[0])]
        match = match.reset_index()
        match = match[['family', 'domain','time']]
        match = match.rename(columns={"domain": "domain_match", "time": "last_seen", "family": "family_match"})
        intel_match = answer.join(match)
        df = df.join(intel_match)
        df = df.set_index(0).T
        return render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values)


@app.route('/contact', methods=['POST', 'GET'])
def contact():
    if current_user.is_authenticated:
        user_id = current_user.id
    else:
        user_id = None
    record = SiteRequest(user_id=user_id, ip_address=request.remote_addr, page=request.path)
    db.session.add(record)
    db.session.commit()
    form = ContactForm()

        # return flask.redirect('authorize')
        # Load credentials from the session.
    credentials = service_account.Credentials.from_service_account_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES).with_subject('contact@blodgic.com')
    gmail = build('gmail', 'v1', credentials=credentials)

    if form.validate_on_submit():
        msg = MIMEText(form.message.data)
        msg['to'] = "contact@blodgic.com"
        msg['subject'] = form.email.data + ': ' + form.subject.data
        # try:
        gmail.users().messages().send(userId="me", body={'raw': urlsafe_b64encode(msg.as_bytes()).decode()}).execute()
        return jsonify({'success': 'Sent!'})

    return render_template('contact.html', title='Contact', form=form)


@app.route('/terms', methods=['GET'])
def terms():
    if current_user.is_authenticated:
        user_id = current_user.id
    else:
        user_id = None
    record = SiteRequest(user_id=user_id, ip_address=request.remote_addr, page=request.path)
    db.session.add(record)
    db.session.commit()
    return render_template('terms.html', title='Terms of Service')

@app.route('/about', methods=['GET'])
def about(): 
    if current_user.is_authenticated:
        user_id = current_user.id
    else:
        user_id = None
    record = SiteRequest(user_id=user_id, ip_address=request.remote_addr, page=request.path)
    db.session.add(record)
    db.session.commit()
    return render_template('about.html', title='About')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        user_id = current_user.id
        return redirect(url_for('index'))
    else:
        user_id = None
    record = SiteRequest(user_id=user_id, ip_address=request.remote_addr, page=request.path)
    db.session.add(record)
    db.session.commit()
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout', methods=['GET'])
def logout_view():
    if current_user.is_authenticated:
        user_id = current_user.id
    else:
        user_id = None
    record = SiteRequest(user_id=user_id, ip_address=request.remote_addr, page=request.path)
    db.session.add(record)
    db.session.commit()
    logout_user()
    return redirect(url_for('index'))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)


@app.route('/api_page', methods=['GET', 'POST'])
def api_page():
    if request.method == 'GET':
        if current_user.is_authenticated:
            user_id = current_user.id
        else:
            user_id = None
        record = SiteRequest(user_id=user_id, ip_address=request.remote_addr, page=request.path)
        db.session.add(record)
        db.session.commit()
        return render_template('api_page.html', title='API')


@app.route('/api', methods=['POST'])
@token_auth.login_required
def api():
    if request.method == 'POST':
        token = APIToken.query.filter_by(token=request.headers.get('Authorization')[7:]).first()
        if current_user.is_authenticated:
            record = APIRequest(user_id=current_user.id, ip_address=request.remote_addr, query=request.get_json()['domain'], token=token.id)
        else:
            record = APIRequest(user_id=None, ip_address=request.remote_addr, query=request.get_json()['domain'], token=token.id)
        db.session.add(record)
        db.session.commit()
        result = request.get_json()
        df = pd.DataFrame.from_dict(result, orient='index')
        #transform weird host entries
        df[0] = df[0].str.lower()
        df[0] = df[0].str.replace(r'(^hxxps?\:\/\/)', '')
        df[0] = df[0].str.replace(r'(^https?\:\/\/)', '')
        df[0] = df[0].str.replace(r'(^www)','')
        df[0] = df[0].str.replace(r'(^www\d+)','')
        df[0] = df[0].str.replace(r'([\[|\]])','')
        df[0] = df[0].str.split(r'\/').str[0]
        df[0] = df[0].str.replace(r'^\.','')

        #entropy
        def entropy(string):
                            #get prob of chars in string
                            prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]

                            #calculate the entropy
                            entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])

                            return entropy

        df['entropy'] = df[0].apply(entropy)

        #ip address
        ippattern = (r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
        df['IP'] = df[0].str.contains(ippattern)

        #tld pattern
        tldpattern = r'\.([^.\n\s]\w+)$'
        df['tld'] = df[0].str.findall(tldpattern).astype(str)
        df['tld'] = df['tld'].replace('[\[|\]|\/|\.|\']', '', regex=True, inplace=False)
        df['tld'] = df['tld'].str.split(',').str[0]

        #hyphen count
        df['hyphen_count'] = df[0].str.count('-')
        #dot count
        df['dot_count'] = df[0].str.count(r'\.')
        #string length of query1
        df['string_len_query1'] = df[0].str.len()
        #tld length
        df['tld_len'] = df['tld'].str.len()

        #count of vowels and consonents
        vowels = set("aeiou")
        cons = set("bcdfghjklmnpqrstvwxyz")
        df['Vowels'] = [sum(1 for c in x if c in vowels) for x in df[0]]
        df['Consonents'] = [sum(1 for c in x if c in cons) for x in df[0]]

        #count the number of syllables in a word
        import re
        def syllables(word):
            word = word.lower()
            if word.endswith('e'):
                word = word[:-1]
            count = len(re.findall('[aeiou]+', word))
            return count

        df['syllables'] = df[0].apply(syllables)

        #count the number of digets in domain
        df['digit_count'] = df[0].apply(lambda x: len([s for s in x if s.isdigit()]))

        #extract host, domain, subdomains
        df['subdomain'] = df[0].apply(lambda url: tldextract.extract(url).subdomain)
        df['domain'] = df[0].apply(lambda url: tldextract.extract(url).domain)
        df['host'] = df[['subdomain', 'domain']].apply(lambda x: '.'.join(x), axis=1)
        df['host'] = df['host'].str.replace(r'^\.', '').astype(str)

        #consonents to vowels ratio
        df['consec_vowel_ratio'] = (df['Vowels'] / df['Consonents']).round(5)

        #count the longest string without vowels
        from itertools import groupby
        is_vowel = lambda char: char in r'aeious0123456789\:\.\-\\'

        def suiteConsonnes(in_str):
            return ["".join(g) for v, g in groupby(in_str, key=is_vowel) if not v]

        string_len = df[0].apply(suiteConsonnes)
        string_len = pd.DataFrame(string_len).reset_index()
        string_len.columns = ['index', 'consec_cons']

        def longestconc(word):
            for letter in word:
                return max(word, key=len)

        string_len['longest_conc'] = string_len['consec_cons'].apply(longestconc)
        string_len['longest_conc_count'] = string_len['longest_conc'].str.len()
        string_len['longest_conc_count'] = string_len['longest_conc_count'].fillna(0)
        string_len = pd.DataFrame(string_len)

        df['consec_vowel_ratio'] = df['consec_vowel_ratio'].fillna(0)

        #move dependent variable (good_bad) to end
        #cols = list(df.columns.values)
        #cols.pop(cols.index('good_bad'))
        #df = df[cols+['good_bad']]

        df = df.reset_index()
        df = df.merge(string_len, on='index')

        #replace long_conc_count if over 10
        #df['long_conc_count'] = df['long_conc_count'].fillna(0)

        #count digits and alpha (letters) in domain
        def count_digits(string):
            return sum(item.isdigit() for item in string)

        def count_words(string):
            return sum(item.isalpha() for item in string)

        df['alpha_count'] = df[0].apply(count_words)
        df['word_digit_ratio'] = (df['digit_count'] / df['alpha_count']).round(5)
        #print('begning wordsegment')
        #wordsegment_start = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
        #print('wordsegment start')
        #print(wordsegment_start)
        from wordsegment import load, segment, clean
        load()
        df['wordsegment_host'] = df.host.apply(segment)
        #count the elements in wordsegment_host list
        df['wordsegment_host_count'] = df['wordsegment_host'].map(len)
        wordsegment_finish = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
        #print('*wordsegment finish*:')
        #print(wordsegment_finish)

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

        df['unique_char_count'] = df['host'].apply(char_frequency).map(len)
        print('domain tried: ')
        print(df.iloc[0])
        #normalize the data
        #replace info for features
        df.consec_vowel_ratio = df.consec_vowel_ratio.replace([np.inf, -np.inf], 0)
        df.entropy = df.entropy.replace([np.inf, -np.inf], 0)
        df.longest_conc_count = df.longest_conc_count.fillna(0)
        df.consec_vowel_ratio = df.consec_vowel_ratio.fillna(0)
        df.word_digit_ratio = df.word_digit_ratio.fillna(0)
        df.word_digit_ratio = df.word_digit_ratio.replace([np.inf, -np.inf], 0)

        #move dependent variable (good_bad) to end
        #cols = list(df.columns.values)
        #cols.pop(cols.index('good_bad'))
        #df = df[cols+['good_bad']]

        #add training features and split to training/test
        #one hot for tld
        testme = df
        onehot = pd.get_dummies(df, columns=['tld',], prefix=['tld'])
        X_predict2 = onehot.filter(['entropy', 'hyphen_count', 'string_len_query1', 'Vowels', 'Consonents',
                                   'syllables', 'digit_count', 'consec_vowel_ratio', 'dot_count', 'longest_conc_count'
                                   'word_digit_ratio', 'tld_len', 'wordsegment_host_count','unique_char_count'])

        #pickup and load model against features
        if app.config['ENV'] == 'development':
            filename = ('../algo/xgboost_domain_binary.sav')
            filename2 = '../algo/multiclass_xgboost.sav'
        else:
            filename = ('/home/ubuntu/python/DGA/algo/xgboost_domain_binary.sav')
            filename2 = '/home/ubuntu/python/DGA/algo/multiclass_xgboost.sav'
        loaded_model = joblib.load(filename)
        result = loaded_model.predict_proba(X_predict2)
        results = pd.DataFrame(result)
        results.columns = ['benign', 'DGA']
        algo = testme.join(results)
        algo_url = algo[['benign', 'DGA']].copy()
        df = df.join(algo_url)
        loaded_model2 = joblib.load(filename2)
        result_class = loaded_model2.predict(X_predict2)
        result_class = pd.DataFrame(result_class)
        result_class.columns = ['DGA_family']
        df = df.join(result_class)
        df.loc[(df['DGA'] >= 0.5) & (df['DGA_family'] == 'benign'), 'DGA_family'] = 'unknown'
        #import DGA intel csv
        if app.config['ENV'] == 'development':
            domain_dga = pd.read_csv('../data/domain_dga.csv')
        else:
            domain_dga = pd.read_csv('/home/ubuntu/python/DGA/data/domain_dga.csv')
        domain_dga['domain'] = domain_dga['domain'].astype(str)
        #the match function to bring back all rows
        answer = df[0].isin(domain_dga['domain'])
        answer = pd.DataFrame(answer)
        answer.columns = ['DGA_match']
        match = domain_dga.loc[domain_dga['domain'].isin(df[0])]
        match = match.reset_index()
        match = match[['family', 'domain','time']]
        match = match.rename(columns={"domain": "domain_match", "time": "last_seen", "family": "family_match"})
        intel_match = answer.join(match)
        df = df.join(intel_match)
        df = df.set_index(0).T
        return app.response_class(response=df.to_json(), status=200, mimetype='application/json')



app.register_blueprint(authtoken)
# app.register_blueprint(email_api)
if app.config['ENV'] == 'development':
    admin = Admin(app, name='Blodgic', template_mode='bootstrap3', index_view=BlodgicAdminIndexView())
    admin.add_view(UserView(User, db.session, 'Users'))
    admin.add_view(SiteRequestView(SiteRequest, db.session, 'Site Visits'))
    admin.add_view(APIRequestView(APIRequest, db.session, 'API Requests', category='API'))
    admin.add_view(APITokenView(APIToken, db.session, 'API Tokens', category='API'))
    admin.add_view(DomainDGAView(DomainDGA, db.session, 'DGA Domains', category='Domains'))
    admin.add_view(TopDomainView(TopDomain, db.session, 'Benign Domains', category='Domains'))
    admin.add_link(MenuLink(name='Home Page', url='/'))


if __name__ == '__main__':
    logging.basicConfig(filename='app.log', level=logging.DEBUG)
    app.run('0.0.0.0', 5000, debug=True, threaded=True)


            








