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

from misc import save_model, load_model, regression_results, grid_search_cv

# +
# Options of settings with different Xs and Ys 
options = ["../data/Train_Drug_Viral_interactions_with_LS_v2_for_Supervised_Learning.csv",
           ".."] #(to be continued)

data_type_options = ["_LS_Drug_LS_Protein",
                     ".."
                    ]

# input option is also used to control the model parameters below
input_option = 0

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
## XGBoost Algorithm
if classification_task:
    model = xgb.XGBClassifier(random_state=0, n_jobs=-1, objective="binary:logistic")
else:
    model = xgb.XGBRegressor(random_state=0, n_jobs=-1, objective="reg:squarederror")
    
# Grid parameters
param_xgb = {"n_estimators": scipy.stats.randint(20, 500),
               "max_depth": scipy.stats.randint(1, 9),
               "gamma": loguniform(1e-8, 1.0),
               "min_child_weight": scipy.stats.randint(1, 10),
               "learning_rate": loguniform(1e-4, 1e-1),
               "subsample": loguniform(0.8, 1e0),
               "reg_alpha": loguniform(1e-1, 1e1),
               "reg_lambda": loguniform(1, 1e1),
}   

n_iter=200

if classification_task:
    xgb_gs=supervised_learning_steps("xgb","roc_auc",data_type,classification_task,model,param_xgb,X_train,y_train,n_iter)
else:
    xgb_gs=supervised_learning_steps("xgb","r2",data_type,classification_task,model,param_xgb,X_train,y_train,n_iter)

xgb_gs.cv_results_
# -

xgb_gs = load_model("xgb_models/xgb__LS_Drug_LS_Protein_regressor_gs.pk")
xgb_best = xgb_gs.best_estimator_
y_pred_xgb=xgb_best.predict(X_train)
plt.hist(y_pred_xgb)
calculate_regression_metrics(y_train,y_pred_xgb)

# +
np.max(xgb_gs.cv_results_["mean_test_score"])
filename = "../data/Test_Drug_Viral_interactions_with_LS_v2_for_Supervised_Learning.csv"
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
y_pred_xgb=xgb_best.predict(X_test)
calculate_regression_metrics(y_test,y_pred_xgb)
# -

meta_X.loc[:,'predictions']=y_pred_xgb
meta_X.loc[:,'labels']=y_test
rev_output_df = meta_X.iloc[:,[0,2,4,5]].copy()
rev_output_df.to_csv("../results/XGB_supervised_test_predictions.csv",index=False)

# +
## load JS visualization code to notebook (Doesn't work for random forest)
#shap.initjs()

## explain the model's predictions using SHAP values
#explainer = shap.TreeExplainer(xgb_gs.best_estimator_)
#shap_values = explainer.shap_values(X_train)
#shap.summary_plot(shap_values, X_train)
# +
#Get results for SARS-COV-2
big_X_test = pd.read_csv("../data/COVID-19/sars_cov_2_drug_viral_interactions_to_predict_with_LS_v2.csv",header='infer',sep=",")
total_length = len(big_X_test.columns)
X_test = big_X_test.iloc[:,range(8,total_length)]
xgb_best = load_model("../models/xgb_models/xgb__LS_Drug_LS_Protein_regressor_best_estimator.pk")
y_pred = xgb_best.predict(X_test)

meta_X_test = big_X_test.iloc[:,[0,2]].copy()
meta_X_test.loc[:,'predictions']=y_pred
meta_X_test.loc[:,'labels']=0
meta_X_test.to_csv("../results/XGB_supervised_sars_cov2_test_predictions.csv",index=False)
# -


