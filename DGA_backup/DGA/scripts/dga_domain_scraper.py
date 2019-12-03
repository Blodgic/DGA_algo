#`/usr/bin/env python3
import requests 
import pandas as pd 
import sys 
from datetime import datetime 
import json

print('Starting Netlab Scrape')
start_d= datetime.now()
start = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
print(r"""
            %s
            
        """  % (start))

netlab_url = 'http://data.netlab.360.com/feeds/dga/dga.txt'
try:
    response = requests.get(netlab_url, stream= True).content
    #response.raise_for_status()
    netlab = str(response).split('\\n')
    netlab = pd.DataFrame(netlab)[18:]
    netlab[0] = netlab[0].astype(str)
    netlab = netlab[0].str.split(r'\\t', expand=True)
    netlab.columns = ['family', 'domain', 'time', 'time2']
    netlab = netlab[['family', 'domain', 'time']]
    netlab['source'] = 'netlab'
    print("Length of Netlab domains")
    print(len(netlab))
    with open('../data/domain_dga.csv','a') as fd:
        netlab.to_csv(fd, index=False, header=True)
#except (http.client.IncompleteRead) as e:
    #page = e.partial
except requests.exceptions.HTTPError as err:
    print(err)
#except requests.exceptions.RequestException as e:  # This is the correct syntax
    #print(e)
    #sys.exit(1)
print('Starting osint scrape')
some_url = 'http://osint.bambenekconsulting.com/feeds/dga-feed.txt'
try:
    response = requests.get(some_url).content
    osint = str(response).split('\\n')
    osint = osint[14:]
    osint = pd.DataFrame(osint)
    osint = osint[0].str.split(',', expand=True)
    osint.columns = ['domain', 'family', 'time', 'source']
    family = '(?:Domain\s+used\s+by\s+)(\w+)'
    osint['family'] = osint['family'].str.findall(family)
    osint['family'] = osint['family'].str[0]
    osint = osint[['family','domain','time','source']].copy()
    print('length of osint')
    print(len(osint))
    with open('../data/domain_dga.csv','a') as fd:
        osint.to_csv(fd, index=False, header=True)    
except requests.exceptions.HTTPError as err:
    print(err)

print('starting dgarchive')
url = 'https://dgarchive.caad.fkie.fraunhofer.de/today'

try:
    response = requests.get(url, stream=True, auth=('lodge', 'digbunchcarrybarretore'))
    data = response.json()
    df = pd.DataFrame.from_dict(data, orient='index')
    df_t = df.transpose()
    dgarchive = df_t.stack().reset_index()
    dgarchive.columns = ['index','family','domain']
    time = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
    dgarchive['time'] = time
    dgarchive['source'] = 'dgarchive'
    dgarchive['family'] = dgarchive.family.str.replace(r'\_.*','a').astype(str)
    dgarchive = dgarchive.drop(columns=['index'])
    print('len of dgarchive')
    print(len(dgarchive))
    with open('../data/domain_dga.csv','a') as fd:
        dgarchive.to_csv(fd, index=False, header=True)
except requests.exceptions.HTTPError as err:
    print(err)

print('length of netlab + osint + dgarchive')
netlab_len = len(netlab)
osint_len = len(osint)
dgarchive_len = len(dgarchive)
print(netlab_len + osint_len + dgarchive_len)

print('removing duplicates')
filename = '../data/domain_dga.csv'
df = pd.read_csv(filename, sep=',')
df.columns = ['family', 'domain', 'time', 'source']
df.domain = df.domain.astype(str)
df['domain'] = df['domain'].str.lower()
df = df.drop_duplicates(['domain'], keep='last')
df = df[:-1]
today = datetime.now()
today = today.strftime('%Y%m%d %H:%M:%S')
print(today)
print('Count of DGAs')
print(len(df))

with open('../data/domain_dga.csv', 'w') as fd:
    df.to_csv(fd, index=False, header=True)

import datetime 
from datetime import datetime

print('END TIME')
ENDTIME_d = datetime.now()
ENDTIME = datetime.strftime(datetime.now(), '%Y%m%d %H:%M:%S')
print(ENDTIME)

import datetime
from datetime import datetime
print('RUN TIME')
diff = ENDTIME_d - start_d
print('****************')
print(diff)


