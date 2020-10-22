# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import re 

import xgboost as xgb
import shap

from sklearn import ensemble
from sklearn import dummy
from sklearn import linear_model
from sklearn import svm
from sklearn import neural_network
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import loguniform
import scipy
import argparse

from misc import save_model, load_model, regression_results, grid_search_cv

# +
def calculate_classification_metrics(labels, predictions):
    
    predictions = predictions.round()
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(labels,predictions)
    
    return metrics.accuracy_score(labels, predictions),\
            metrics.f1_score(labels, predictions, average='binary'),\
            auc,\
            aupr


def calculate_regression_metrics(labels, predictions):
    return metrics.mean_absolute_error(labels, predictions),\
            metrics.mean_squared_error(labels, predictions),\
            metrics.r2_score(labels, predictions),\
            scipy.stats.pearsonr(np.array(labels).flatten(),np.array(predictions.flatten()))[0],\
            scipy.stats.spearmanr(np.array(labels).flatten(),np.array(predictions.flatten()))[0]


# -

def supervised_learning_steps(method,scoring,data_type,task,model,params,X_train,y_train,n_iter):
    
    gs = grid_search_cv(model, params, X_train, y_train, scoring=scoring, n_iter = n_iter)

    y_pred = gs.predict(X_train)
    y_pred[y_pred < 0] = 0

    if task:
        results=calculate_classification_metrics(y_train, y_pred)
        print("Acc: %.3f, F1: %.3f, AUC: %.3f, AUPR: %.3f" % (results[0], results[1], results[2], results[3]))
    else:
        results=calculate_regression_metrics(y_train,y_pred)
        print("MAE: %.3f, MSE: %.3f, R2: %.3f, Pearson R: %.3f, Spearman R: %.3f" % (results[0], results[1], results[2], results[3], results[4]))
   
    print('Parameters')
    print('----------')
    for p,v in gs.best_estimator_.get_params().items():
        print(p, ":", v)
    print('-' * 80)

    if task:
        save_model(gs, "%s_models/%s_%s_classifier_gs.pk" % (method,method,data_type))
        save_model(gs.best_estimator_, "%s_models/%s_%s_classifier_best_estimator.pk" %(method,method,data_type))
    else:
        save_model(gs, "%s_models/%s_%s_regressor_gs.pk" % (method,method,data_type))
        save_model(gs.best_estimator_, "%s_models/%s_%s_regressor_best_estimator.pk" %(method,method,data_type))
        
    return(gs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions on test set of compound viral protein interaction pairs using GLM method')
    parser.add_argument('input1', help='Train compound-viral protein activities file containing latent space representations for compounds and proteins')
    parser.add_argument('input2', help='Test compound-viral protein pairs file containing latent space representations for compounds and proteins')
    parser.add_argument('output', help='Output of prediction of GLM method for the test file')
    args = parser.parse_args()



    # +
    # Options of settings with different Xs and Ys 
    #options = ["../data/Train_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv",
    #           "../data/Train_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv",
    #           ".."] #(to be continued)

    #data_type_options = ["LS_Compound_LS_Protein",
    #                     "MFP_Compound_LS_Protein",
    #                     ".."
    #                    ]

    # input option is also used to control the model parameters below
    #input_option = 0

    #classification_task = False
    #classification_th = 85

    #data_type=data_type_options[input_option]
    filename = "../data/"+args.input1

    with open(filename, "rb") as file:
        print("Loading Training data: ", filename)
        big_df = pd.read_csv(filename, header='infer', delimiter=",")
        total_length = len(big_df.columns)
        X = big_df.iloc[:,range(5,total_length)]
        Y = big_df[['pchembl_value']].to_numpy().flatten()
        meta_X = big_df.iloc[:,[0,1,2,3]]
        print("Lengths --> X = %d, Y = %d" % (len(X), len(Y)))

    print(X.columns)
    n_samples = len(X)
    indices = np.arange(n_samples)

    X_train = X
    y_train = Y
    print(X_train[:10])
    print(X_train.shape,y_train.shape)
    print(X_train.columns)
    print(big_df.isnull().sum().sum())
    print("Loaded training file")

    # +
    '''
    #Build the Generalized Linear Regression model
    if classification_task:
        model = linear_model.LogisticRegression()
    else:
        model = linear_model.LinearRegression()

    # Grid parameters
    if classification_task:
        params_lr = [
           {
                'C': scipy.stats.randint(20, 500),
                'fit_intercept': [True, False],
           }
        ]
    else:
        params_lr = [
           {
                'normalize': [True, False],
                'fit_intercept': [True, False],
           }
        ]

    n_iter = 1500
    scaler = preprocessing.MinMaxScaler()
    X_train_copy = scaler.fit_transform(X_train)

    if classification_task:
        lr_gs=supervised_learning_steps("glm","roc_auc",data_type,classification_task,model,params_lr,X_train_copy,y_train,n_iter)
    else:
        lr_gs=supervised_learning_steps("glm","r2",data_type,classification_task,model,params_lr,X_train_copy,y_train,n_iter)
        
    print(lr_gs.cv_results_)
    save_model(scaler, "%s_models/%s_%s_scaling_gs.pk" % ("glm","glm",data_type))


    # +
    #Test the generalized linear regression model on separate test set
    np.max(lr_gs.cv_results_["mean_test_score"])

    file_list = ["../data/Test_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv",
                 "../data/Test_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv"]

    filename = file_list[input_option]

    with open(filename, "rb") as file:
        print("Loading ", filename)
        big_df = pd.read_csv(filename, header='infer', delimiter=",")
        total_length = len(big_df.columns)
        X = big_df.iloc[:,range(5,total_length)]
        Y = big_df[['pchembl_value']].to_numpy().flatten()
        meta_X = big_df.iloc[:,[0,1,2,3]]
        print("Lengths --> X = %d, Y = %d" % (len(X), len(Y)))

    print(X.columns)
    n_samples = len(X)
    indices = np.arange(n_samples)

    X_test = X
    y_test = Y
    lr_best = lr_gs.best_estimator_
    y_pred_lr=lr_best.predict(scaler.transform(X_test))
    print(calculate_regression_metrics(y_test,y_pred_lr))

    #Write the prediction of GLM model
    meta_X["predictions"]=y_pred_lr
    meta_X["labels"]=y_test
    rev_output_df = meta_X.iloc[:,[0,2,4,5]].copy()
    rev_output_df.to_csv("../results/GLM_"+data_type_options[input_option]+"_supervised_test_predictions.csv",index=False)

    # +
    #Dummy mean regressor and median regressor
    strategy = 'mean'
    model = dummy.DummyRegressor(strategy=strategy)
    model.fit(X_train,y_train)
    y_pred_mean = model.predict(X_test)
    print(calculate_regression_metrics(y_test,y_pred_mean))

    strategy = 'median'
    model = dummy.DummyRegressor(strategy=strategy)
    model.fit(X_train,y_train)
    y_pred_median = model.predict(X_test)
    print(calculate_regression_metrics(y_test,y_pred_median))
    # -
    '''

    #Get results for test file
    print("Loading test file")
    test_filename = args.input2
    big_X_test = pd.read_csv("../data/"+args.input2,header='infer',sep=",")
    total_length = len(big_X_test.columns)
    X_test = big_X_test.iloc[:,range(5,total_length)]
    
    if ("MFP" not in args.input1):
        lr_best = load_model("../models/glm_models/glm_LS_Compound_LS_Protein_regressor_best_estimator.pk")
        scaler = load_model("../models/glm_models/glm_LS_Compound_LS_Protein_scaling_gs.pk")
    else:
        lr_best = load_model("../models/glm_models/glm_MFP_Compound_LS_Protein_regressor_best_estimator.pk")
        scaler = load_model("../models/glm_models/glm_MFP_Compound_LS_Protein_scaling_gs.pk")
    
    print("Making Predictions")
    y_pred = lr_best.predict(scaler.transform(X_test))

    meta_X_test = big_X_test.iloc[:,[0,2]].copy()
    meta_X_test.loc[:,'predictions']=y_pred
    if ("sars_cov_2" in args.input2):
        meta_X_test.loc[:,'labels']=0
    else:
        meta_X_test.loc[:,'labels']=big_X_test.iloc[:,4].copy()

    out_file = args.output
    meta_X_test.to_csv("../results/"+out_file,index=False)
    print("Finished writing predictions")

