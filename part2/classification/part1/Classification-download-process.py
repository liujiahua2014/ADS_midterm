
# coding: utf-8

# In[1]:

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from collections import OrderedDict
from random import randint, random
from numbers import Number


# In[2]:

def download_file(url, local_filename):
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


# In[3]:

basicUrl = 'https://freddiemac.embs.com'
basicSecureUrl = basicUrl + '/FLoan/secure/'
loginPageUrl = basicSecureUrl + 'login.php'


# In[4]:

with requests.session() as s:
    # Login Page Get
    request = s.get(loginPageUrl)
    loginPageSoup = BeautifulSoup(request.text, 'lxml')
    
    # Login Post
    loginPostUrl = basicSecureUrl + loginPageSoup.find_all('form')[0]['action']
    payload = {}
    payload['username'] = 'liu.jiah@husky.neu.edu'
    payload['password'] = 'AK~JihFG'
    response = s.post(loginPostUrl, data=payload)
    TandCSoup = BeautifulSoup(response.text, 'lxml')

    # T and C Post
    downloadUrl = basicUrl + TandCSoup.find_all('form')[0]['action']
    payload = {}
    payload['accept'] = 'Yes'
    payload['acceptSubmit'] = 'Continue'
    payload['action'] = 'acceptTandC'
    response = s.post(downloadUrl, data=payload)
    downloadPageSoup = BeautifulSoup(response.text, 'lxml')


# In[ ]:

filenames = []

# Download Files Get
for elem in downloadPageSoup.find_all('a'):
    filename = elem.get_text()
    url = 'https://freddiemac.embs.com/FLoan/Data/' + filename
    year = filename[-8:-4]
    quarter = filename[-9:-4]
    if filename.startswith('historical') and int(year) == 2005 and int(quarter) in [1, 2]:
        filenames.append(filename)
        download_file(url, filename)


# In[7]:

import zipfile
import os


# In[8]:

# Unzip file
for file in filenames:
    with zipfile.ZipFile(file,"r") as zip_ref:
        zip_ref.extractall(file.split('.')[0])
        zip_ref.close()
    os.remove(file)


# In[ ]:




# In[10]:

def calMean(df, col):
    std = df[col].std()
    mean = df[col].mean()
    left = mean - 1.5 * std
    right = mean + 1.5 * std
    realMean = df[(df[col] > left) & (df[col] < right)][col].mean()
    return realMean


# In[11]:

def cleanOrigData(orig_file, clean_file):
    orig_col_headers = ['credit_score', 'first_payment_date', 'first_time_homebuyer_flag', 'maturity_date', 'metropolitan_stat_area', 'mortgage_insurance_perc', 'no_unit', 'occupancy_status', 'orig_combined_loantovalue', 'orig_debttoincome', 'orig_upb', 'orig_loantovalue', 'orig_interest_rate', 'channel', 'prepayment_penalty_mortgage_flag', 'product_type', 'property_state', 'property_type', 'postal_code', 'loan_sequence_no', 'loan_purpose', 'orig_loan_term', 'no_borrower', 'seller_name', 'service_name', 'super_conforming_flag']
    orig_df = pd.read_csv(orig_file, sep='|', names=orig_col_headers)
    
    if not np.issubdtype(orig_df['credit_score'].dtype, np.number):
        orig_df.loc[orig_df['credit_score'] == "   ", 'credit_score'] = 300
    orig_df['credit_score'] = orig_df['credit_score'].astype(int)
    
    if not np.issubdtype(orig_df['orig_debttoincome'].dtype, np.number):
        for ind, row in orig_df.iterrows():
            if row['orig_debttoincome'] == "   ":
                orig_df.iloc[ind, orig_df.columns.get_loc('orig_debttoincome')] = randint(65, 70)
                
    new_df = orig_df[orig_df['orig_debttoincome'].apply(lambda x: isinstance(x, Number))]
    mean = calMean(new_df, 'orig_debttoincome')
    orig_df['orig_debttoincome'] = orig_df['orig_debttoincome'].fillna(mean).astype(int)
    
    orig_df['first_time_homebuyer_flag'] = orig_df['first_time_homebuyer_flag'].fillna('N')
    
    orig_df['metropolitan_stat_area'] = orig_df['metropolitan_stat_area'].fillna(0)
    
    orig_df['no_unit'] = orig_df['no_unit'].fillna(1)
    
    mode = orig_df['orig_combined_loantovalue'].mode()[0]
    new_df = orig_df[orig_df['orig_combined_loantovalue'] != mode]
    mean = calMean(new_df, 'orig_combined_loantovalue')
    orig_df['orig_combined_loantovalue'] = orig_df['orig_combined_loantovalue'].fillna(mean).astype(int)
        
    mode = orig_df['orig_loantovalue'].mode()[0]
    new_df = orig_df[orig_df['orig_loantovalue'] != mode]
    mean = calMean(new_df, 'orig_loantovalue')
    orig_df['orig_loantovalue'] = orig_df['orig_loantovalue'].fillna(mean).astype(int)
    
    orig_df['prepayment_penalty_mortgage_flag'] = orig_df['prepayment_penalty_mortgage_flag'].fillna('N')
    
    orig_df['postal_code'] = orig_df['postal_code'].fillna('00000')
    
    one_borrower_perc = orig_df['no_borrower'].value_counts()[1.0] / orig_df.shape[0]
    temp_df = orig_df[orig_df['no_borrower'] != orig_df['no_borrower']]
    for ind, row in temp_df.iterrows():
        orig_df.iloc[ind, orig_df.columns.get_loc('no_borrower')] = 1 if random() < one_borrower_perc else 2
    
    if not np.issubdtype(orig_df['mortgage_insurance_perc'].dtype, np.number):
        orig_df.loc[orig_df['mortgage_insurance_perc'] == "   ", 'mortgage_insurance_perc'] = 0
        orig_df.loc[orig_df['mortgage_insurance_perc'] == "000", 'mortgage_insurance_perc'] = 0
    
    orig_df = orig_df.drop('super_conforming_flag', 1)
    
    orig_df.to_csv(clean_file, index=False)

    return orig_df


# In[12]:

def cleanPerfData(perf_file, clean_file):
    perform_col_headers = ['loan_sequence_no', 'monthly_reporting_period', 'curr_actual_upb', 'curr_loan_delinquency_status', 'loan_age', 'remaining_months_to_legal_maturity', 'repurchase_flag', 'modification_flag', 'zero_balance_code', 'zero_balance_effective_date', 'curr_interest_rate', 'curr_deferred_upb', 'due_date_last_paid_installment', 'mi_recoveries', 'net_sales_proceeds', 'non_mi_recoveries', 'expenses', 'legal_costs', 'maintain_preserve_costs', 'tax_insurance', 'miscellaneous_expense', 'actual_loss_calculation', 'modification_cost']
    perf_df = pd.read_csv(perf_file, sep='|', names=perform_col_headers)
    
    missing_df = perf_df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['col', 'missing_cnt']
    missing_df = missing_df[missing_df['missing_cnt'] > 0]
    
    colsToDrop = missing_df['col'].tolist()
    colsToDrop.remove('actual_loss_calculation')
    perf_df = perf_df.drop(colsToDrop, axis=1)
    
    perf_df.to_csv(clean_file, index=False)

    return perf_df


# In[ ]:




# Clean

# In[ ]:

for year in range(2005, 2005):
    for quarter in range(1, 3):
        orig_file = 'historical_data1_Q{0}{1}/historical_data1_Q{0}{1}.txt'.format(quarter, year)
        orig_clean_file = 'historical_data1_Q{0}{1}/historical_data1_Q{0}{1}_clean.csv'.format(quarter, year)
        orig_df = cleanOrigData(orig_file, orig_clean_file)
        
        perf_file = 'historical_data1_Q{0}{1}/historical_data1_time_Q{0}{1}.txt'.format(quarter, year)
        perf_clean_file = 'historical_data1_Q{0}{1}/historical_data1_time_Q{0}{1}_clean.csv'.format(quarter, year)
        perf_df = cleanPerfData(perf_file, perf_clean_file)

