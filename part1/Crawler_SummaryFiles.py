
# coding: utf-8

# In[64]:

from bs4 import BeautifulSoup
import requests


# In[65]:

def download_file(url, local_filename):
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


# In[66]:

basicUrl = 'https://freddiemac.embs.com'
basicSecureUrl = basicUrl + '/FLoan/secure/'
loginPageUrl = basicSecureUrl + 'login.php'


# In[67]:

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
    
#     print(downloadPageSoup.prettify())


# In[70]:

filenames = []

# Download Files Get
for elem in downloadPageSoup.find_all('a'):
    filename = elem.get_text()
    url = 'https://freddiemac.embs.com/FLoan/Data/' + filename
    year = filename[-8:-4]
#     if filename.startswith('sample') and int(year) == 2005:
    if filename.startswith('historical') and int(year) in [1999, 2013, 2016]:
        filenames.append(filename)
        download_file(url, filename)


# In[71]:

import zipfile
import os


# In[72]:

# Unzip file
for file in filenames:
    with zipfile.ZipFile(file,"r") as zip_ref:
        zip_ref.extractall(file.split('.')[0])
        zip_ref.close()
    os.remove(file)


# In[ ]:




# Create Summary Files

# In[13]:

import pandas as pd

pd.set_option('display.max_columns', None)


# In[14]:

import numpy as np


# In[15]:

from collections import OrderedDict


# In[16]:

from random import randint, random


# In[51]:

from numbers import Number


# In[ ]:



# In[ ]:




# In[ ]:




# In[41]:

def calMean(df, col):
    std = df[col].std()
    mean = df[col].mean()
    left = mean - 1.5 * std
    right = mean + 1.5 * std
    realMean = df[(df[col] > left) & (df[col] < right)][col].mean()
    return realMean


# In[74]:

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


# In[75]:

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




# Create Summary Files

# In[63]:

orig_summary_list = []
orig_summary_state_list = []
perf_summary_list = []

for year in range(2005, 2017):
    
    # Origination Summary File By Year
    
    orig_file = 'sample_{0}/sample_orig_{0}.txt'.format(year)
    orig_clean_file = 'sample_{0}/sample_orig_{0}_clean.csv'.format(year)
    
    orig_df = cleanOrigData(orig_file, orig_clean_file)
 
    orig_summary_dict = OrderedDict()
    
    orig_summary_dict['year'] = year
    
    loanCount = np.count_nonzero(orig_df['loan_sequence_no'])
    orig_summary_dict['loanCount'] = loanCount
    
    totalOrigUPB = orig_df['orig_upb'].sum()
    orig_summary_dict['totalOrigUPB'] = totalOrigUPB
    
    avgOrigUPB = orig_df['orig_upb'].mean()
    orig_summary_dict['avgOrigUPB'] = avgOrigUPB
    
    avgCreditScore = orig_df[orig_df['credit_score'] > 0]['credit_score'].mean()
    orig_summary_dict['avgCreditScore'] = avgCreditScore
    
    avgOrigInterestRate = orig_df['orig_interest_rate'].mean()
    orig_summary_dict['avgOrigInterestRate'] = avgOrigInterestRate
    
    avgOrigCombinedLoantovalue = orig_df['orig_combined_loantovalue'].mean()
    orig_summary_dict['avgOrigCombinedLoantovalue'] = avgOrigCombinedLoantovalue
    
    avgOrigLoantovalue = orig_df['orig_loantovalue'].mean()
    orig_summary_dict['avgOrigLoantovalue'] = avgOrigLoantovalue
    
    avgOrigDebttoincome = orig_df['orig_debttoincome'].mean()
    orig_summary_dict['avgOrigDebttoincome'] = avgOrigDebttoincome
    
    orig_summary_list.append(orig_summary_dict)
    
    # Origination Summary File By State
    
    for state in orig_df['property_state'].unique():
        
        orig_summary_state_dict = OrderedDict()
    
        orig_summary_state_dict['year'] = year
    
        orig_summary_state_dict['state'] = state
        
        loanCount = np.count_nonzero(orig_df[orig_df['property_state'] == state]['loan_sequence_no'])
        orig_summary_state_dict['loanCount'] = loanCount
        
        totalOrigUPB = orig_df[orig_df['property_state'] == state]['orig_upb'].sum()
        orig_summary_state_dict['totalOrigUPB'] = totalOrigUPB

        avgOrigUPB = orig_df[orig_df['property_state'] == state]['orig_upb'].mean()
        orig_summary_state_dict['avgOrigUPB'] = avgOrigUPB

        avgCreditScore = orig_df[(orig_df['property_state'] == state) & (orig_df['credit_score'] > 0)]['credit_score'].mean()
        orig_summary_state_dict['avgCreditScore'] = avgCreditScore

        avgOrigInterestRate = orig_df[orig_df['property_state'] == state]['orig_interest_rate'].mean()
        orig_summary_state_dict['avgOrigInterestRate'] = avgOrigInterestRate

        avgOrigCombinedLoantovalue = orig_df[orig_df['property_state'] == state]['orig_combined_loantovalue'].mean()
        orig_summary_state_dict['avgOrigCombinedLoantovalue'] = avgOrigCombinedLoantovalue

        avgOrigLoantovalue = orig_df[orig_df['property_state'] == state]['orig_loantovalue'].mean()
        orig_summary_state_dict['avgOrigLoantovalue'] = avgOrigLoantovalue

        avgOrigDebttoincome = orig_df[orig_df['property_state'] == state]['orig_debttoincome'].mean()
        orig_summary_state_dict['avgOrigDebttoincome'] = avgOrigDebttoincome
        
        orig_summary_state_list.append(orig_summary_state_dict)
    
    # Performance Summary File
    
    perf_file = 'sample_{0}/sample_svcg_{0}.txt'.format(year)
    perf_clean_file = 'sample_{0}/sample_svcg_{0}_clean.csv'.format(year)
    
    perf_df = cleanPerfData(perf_file, perf_clean_file)
    
    perf_summary_dict = OrderedDict()
    
    perf_summary_dict['year'] = year
    
#     loanCount = np.count_nonzero(perf_df['loan_sequence_no'].unique())
#     perf_summary_dict['loanCount'] = loanCount
    
#     totalCurrActualUpb = perf_df['curr_actual_upb'].sum()
#     perf_summary_dict['totalCurrActualUpb'] = totalCurrActualUpb
    
#     avgCurrActualUpb = perf_df['curr_actual_upb'].mean()
#     perf_summary_dict['avgCurrActualUpb'] = avgCurrActualUpb
    
    perf_df['curr_loan_delinquency_status'] = perf_df['curr_loan_delinquency_status'].astype(str)
    nonDelinquencyRatio = perf_df[perf_df['curr_loan_delinquency_status'] == '0']['curr_loan_delinquency_status'].count() / perf_df.shape[0]
    perf_summary_dict['nonDelinquencyRatio'] = nonDelinquencyRatio
    
    interest_rate_df = perf_df.groupby(['loan_sequence_no'])['curr_interest_rate'].mean().reset_index()
    avgCurrInterestRate = interest_rate_df['curr_interest_rate'].mean()
    perf_summary_dict['avgCurrInterestRate'] = avgCurrInterestRate
    
    perf_summary_list.append(perf_summary_dict)
    
orig_summary_df = pd.DataFrame(orig_summary_list)
orig_summary_df.to_csv('orig_summary.csv', index=False)

orig_summary_state_df = pd.DataFrame(orig_summary_state_list)
orig_summary_state_df.to_csv('orig_summary_state.csv', index=False)

perf_summary_df = pd.DataFrame(perf_summary_list)
perf_summary_df.to_csv('perf_summary.csv', index=False)

