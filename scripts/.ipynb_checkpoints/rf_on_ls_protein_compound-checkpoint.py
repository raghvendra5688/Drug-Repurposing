# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

from misc import save_model, load_model, regression_results, grid_search_cv

# +
# Options of settings with different Xs and Ys 
options = ["../data/Train_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv",
           "../data/Train_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv",
           ".."] #(to be continued)

data_type_options = ["LS_Compound_LS_Protein",
                     "MFP_Compound_LS_Protein",
                     ".."
                    ]

# input option is also used to control the model parameters below
input_option = 1

classification_task = False
classification_th = 85

data_type=data_type_options[input_option]
filename = options[input_option]

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

X_train = X
y_train = Y
print(X_train[:10])
print(X_train.shape,y_train.shape)
print(X_train.columns)
print(big_df.isnull().sum().sum())


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


# +
if classification_task:
    model = ensemble.RandomForestRegressor(n_estimators=100, criterion='auc',
                                            max_depth=None, min_samples_split=2,
                                            min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                            max_features='auto', max_leaf_nodes=None,
                                            min_impurity_decrease=0.0, min_impurity_split=None,
                                            bootstrap=True, oob_score=False,
                                            n_jobs=-1, random_state=328, verbose=1,
                                            warm_start=False, ccp_alpha=0.0, max_samples=None)

else:
    model = ensemble.RandomForestRegressor(n_estimators=100, criterion='mse',
                                            max_depth=None, min_samples_split=2,
                                            min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                            max_features='auto', max_leaf_nodes=None,
                                            min_impurity_decrease=0.0, min_impurity_split=None,
                                            bootstrap=True, oob_score=False,
                                            n_jobs=-1, random_state=328, verbose=1,
                                            warm_start=False, ccp_alpha=0.0, max_samples=None)


# Grid parameters
param_rf = {"n_estimators": scipy.stats.randint(20, 500),
               "max_depth": scipy.stats.randint(1, 9),
               "min_samples_leaf": scipy.stats.randint(1, 10),
               "max_features": scipy.stats.uniform.ppf([0.1,0.7])
}

n_iter=300

if classification_task:
    rf_gs=supervised_learning_steps("rf","roc_auc",data_type,classification_task,model,param_rf,X_train,y_train,n_iter)
else:
    rf_gs=supervised_learning_steps("rf","r2",data_type,classification_task,model,param_rf,X_train,y_train,n_iter)

rf_gs.cv_results_

# +
rf_gs = load_model("rf_models/rf_"+data_type_options[input_option]+"_regressor_gs.pk")
np.max(rf_gs.cv_results_["mean_test_score"])

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
rf_best = rf_gs.best_estimator_
y_pred_rf=rf_best.predict(X_test)
print(calculate_regression_metrics(y_test,y_pred_rf))

#Write the output in the results folder
meta_X["predictions"]=y_pred_rf
meta_X["labels"]=y_test
rev_output_df = meta_X.iloc[:,[0,2,4,5]].copy()
rev_output_df.to_csv("../results/RF_"+data_type_options[input_option]+"supervised_test_predictions.csv",index=False)

# +
## load JS visualization code to notebook (Doesn't work for random forest)
#shap.initjs()

## explain the model's predictions using SHAP values
#explainer = shap.TreeExplainer(xgb_gs.best_estimator_)
#shap_values = explainer.shap_values(X_train)
#shap.summary_plot(shap_values, X_train)
# +
##Get results for SARS-COV-2
#big_X_test = pd.read_csv("../data/COVID-19/sars_cov_2_additional_compound_viral_interactions_to_predict_with_LS_v2.csv",header='infer',sep=",")
#total_length = len(big_X_test.columns)
#X_test = big_X_test.iloc[:,range(8,total_length)]
#rf_best = load_model("../models/rf_models/rf__LS_Drug_LS_Protein_regressor_best_estimator.pk")
#y_pred = rf_best.predict(X_test)

#meta_X_test = big_X_test.iloc[:,[0,2]].copy()
#meta_X_test.loc[:,'predictions']=y_pred
#meta_X_test.loc[:,'labels']=0
#meta_X_test.to_csv("../results/RF_supervised_sars_cov2_additional_test_predictions.csv",index=False)
