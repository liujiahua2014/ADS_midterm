import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import itertools
import time
import statsmodels.api as sm
import sys
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from collections import OrderedDict
from multiprocessing import Manager, Pool


def processSubset(feature_set, X, y):
# Fit model on feature_set and calculate RSS
    model = LogisticRegression()
    model = model.fit(X[list(feature_set)], y)
    Score = model.score(X[list(feature_set)], y)
    return {"model":model, "score":Score, "feature":X[list(feature_set)]}

def forward(predictors, X, y):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        model = processSubset(predictors+[p], X, y)
        results.append(model)
        
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['score'].argmax()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    print best_model
    return best_model

def getBestModel(models):
    length = len(models.index)
    bestModel = models.loc[1]
    score = models.loc[1]["score"]
    for i in range (1, length + 1):
        if models.loc[i]["score"] > score:
            bestModel = models.loc[i]
            score = models.loc[i]["score"]
    return bestModel

def getMatrix(conf_mat_logred, csvFile):
    No_of_actual_delq = conf_mat_logred[1][0] + conf_mat_logred[1][1]
    No_of_pred_delq = conf_mat_logred[0][1] + conf_mat_logred[1][1]
    No_of_records = conf_mat_logred[0][1] + conf_mat_logred[1][1] + conf_mat_logred[1][0] + conf_mat_logred[0][0]
    No_of_delq_properly_classified = conf_mat_logred[1][1]
    No_of_nonDelq_improperly_classified_as_delq = conf_mat_logred[0][1]
    all_conf_df = pd.DataFrame(OrderedDict((('Quarter',[csvFile]), 
                                 ('No_of_actual_delq',[No_of_actual_delq]),
                                 ('No_of_pred_delq',[No_of_pred_delq]),
                                 ('No_of_records',[No_of_records]),
                                 ('No_of_delq_properly_classified',[No_of_delq_properly_classified]),
                                 ('No_of_nonDelq_improperly_classified_as_delq',[No_of_nonDelq_improperly_classified_as_delq]))))
    
    #matrix=pd.concat([all_conf_df, matrix],axis=0)
    return all_conf_df

def getPreperation(csvFile):
    global shared_list
    df = pd.read_csv(csvFile)
    y = df['curr_loan_delinquency_status']
    df = df.dropna().drop(['loan_sequence_no', 'monthly_reporting_period', 
                          'curr_loan_delinquency_status'], axis=1).astype('float64')
    X = df

    models2 = pd.DataFrame(columns=["score", "model", "feature"])
    tic = time.time()
    predictors = []

    for i in range(1,len(X.columns)+1):
        models2.loc[i] = forward(predictors, X, y)
        predictors = list(models2.loc[i]["feature"])

    toc = time.time()
    print("Total elapsed time:", (toc-tic), "seconds.")
    print predictors

    best_model = getBestModel(models2)

    print best_model["score"]
    print list(best_model["feature"])

    y_predict = best_model["model"].predict(X[list(best_model["feature"])])
    conf_mat_logred = metrics.confusion_matrix(y, y_predict)
    shared_list.append(getMatrix(conf_mat_logred, csvFile))

if __name__ == '__main__':
    matrix = pd.DataFrame(OrderedDict((('Quarter',[]), 
                                     ('No_of_actual_delq',[]),
                                     ('No_of_pred_delq',[]),
                                     ('No_of_records',[]),
                                     ('No_of_delq_properly_classified',[]),
                                     ('No_of_nonDelq_improperly_classified_as_delq',[]))))
    filelist = ['data/Q11999.csv', 'data/Q12000.csv', 'data/Q12001.csv', 'data/Q12002.csv', 'data/Q12003.csv',
                'data/Q12004.csv', 'data/Q12005.csv', 'data/Q12006.csv', 'data/Q12007.csv', 'data/Q12008.csv', 
                'data/Q21999.csv', 'data/Q22000.csv', 'data/Q22001.csv', 'data/Q22002.csv', 'data/Q22003.csv',
                'data/Q22004.csv', 'data/Q22005.csv', 'data/Q22006.csv', 'data/Q22007.csv', 'data/Q22008.csv',
                'data/Q31999.csv', 'data/Q32000.csv', 'data/Q32001.csv', 'data/Q32002.csv', 'data/Q32003.csv',
                'data/Q32004.csv', 'data/Q32005.csv', 'data/Q32006.csv', 'data/Q32007.csv', 'data/Q32008.csv',
                'data/Q41999.csv', 'data/Q42000.csv', 'data/Q42001.csv', 'data/Q42002.csv', 'data/Q42003.csv',
                'data/Q42004.csv', 'data/Q42005.csv', 'data/Q42006.csv', 'data/Q42007.csv', 'data/Q42008.csv',]
    manager = Manager()
    shared_list = manager.list()
    
    pool = Pool(processes=4)
    pool.map(getPreperation, filelist)
    for mat in shared_list:
        matrix=pd.concat([mat, matrix],axis=0)
        
    matrix.to_csv('matrix_classification.csv', index = False)