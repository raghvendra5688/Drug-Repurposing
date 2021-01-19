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


# -
def get_cv_results(model, index, cv_scores, nfolds):
    for i in range(0,nfolds):
        cv_scores.append(np.round(model.cv_results_["split"+str(i)+"_test_score"][index], nfolds))
    return(cv_scores)


# +
#Get the CV models and optimal scores

#GLM with protein and compound LS
glm_LS_LS = load_model("../models/glm_models/glm_LS_Compound_LS_Protein_regressor_gs.pk")
glm_LS_LS_index = np.argmax(glm_LS_LS.cv_results_["mean_test_score"])
glm_LS_LS_r2_scores = get_cv_results(glm_LS_LS, glm_LS_LS_index, [], 5)

glm_MFP_LS = load_model("../models/glm_models/glm_MFP_Compound_LS_Protein_regressor_gs.pk")
glm_MFP_LS_index = np.argmax(glm_MFP_LS.cv_results_["mean_test_score"])
glm_MFP_LS_r2_scores = get_cv_results(glm_MFP_LS, glm_MFP_LS_index, [], 5)

#RF with protein and compound LS
rf_LS_LS = load_model("../models/rf_models/rf_LS_Compound_LS_Protein_regressor_gs.pk")
rf_LS_LS_index = np.argmax(rf_LS_LS.cv_results_["mean_test_score"])
rf_LS_LS_r2_scores = get_cv_results(rf_LS_LS, rf_LS_LS_index, [], 5)

rf_MFP_LS = load_model("../models/rf_models/rf_MFP_Compound_LS_Protein_regressor_gs.pk")
rf_MFP_LS_index = np.argmax(rf_MFP_LS.cv_results_["mean_test_score"])
rf_MFP_LS_r2_scores = get_cv_results(rf_MFP_LS, rf_MFP_LS_index, [], 5)

#SVM with protein and compound LS
svm_LS_LS = load_model("../models/svm_models/svm_LS_Compound_LS_Protein_regressor_gs.pk")
svm_LS_LS_index = np.argmax(svm_LS_LS.cv_results_["mean_test_score"])
svm_LS_LS_r2_scores = get_cv_results(svm_LS_LS, svm_LS_LS_index, [], 5)

svm_MFP_LS = load_model("../models/svm_models/svm_MFP_Compound_LS_Protein_regressor_gs.pk")
svm_MFP_LS_index = np.argmax(svm_MFP_LS.cv_results_["mean_test_score"])
svm_MFP_LS_r2_scores = get_cv_results(svm_MFP_LS, svm_MFP_LS_index, [], 5)

#XGB with protein and compound LS
xgb_LS_LS = load_model("../models/xgb_models/xgb_LS_Compound_LS_Protein_regressor_gs.pk")
xgb_LS_LS_index = np.argmax(xgb_LS_LS.cv_results_["mean_test_score"])
xgb_LS_LS_r2_scores = get_cv_results(xgb_LS_LS, xgb_LS_LS_index, [], 5)

xgb_MFP_LS = load_model("../models/xgb_models/xgb_MFP_Compound_LS_Protein_regressor_gs.pk")
xgb_MFP_LS_index = np.argmax(xgb_MFP_LS.cv_results_["mean_test_score"])
xgb_MFP_LS_r2_scores = get_cv_results(xgb_MFP_LS, xgb_MFP_LS_index, [], 5)
# -


all_info = [glm_LS_LS_r2_scores, glm_MFP_LS_r2_scores, rf_LS_LS_r2_scores, rf_MFP_LS_r2_scores,
            svm_LS_LS_r2_scores, svm_MFP_LS_r2_scores, xgb_LS_LS_r2_scores, xgb_MFP_LS_r2_scores]
cv_df = pd.DataFrame(all_info)
cv_df = cv_df.transpose()
cv_df.columns = ["GLM_LS_LS","GLM_MFP_LS","RF_LS_LS","RF_MFP_LS",
                                                       "SVM_LS_LS","SVM_MFP_LS","XGB_LS_LS","XGB_MFP_LS"]



# +
f = plt.figure()
plt.rcParams["figure.figsize"] = (15,15)
cv_df.plot(kind='box')
plt.xlabel('Methods', fontsize=24)
plt.ylabel('R2',fontsize=24)
plt.suptitle("Comparison of CV performance of ML methods",fontsize=28)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

plt.savefig("../results/CV_Comparison.pdf", bbox_inches='tight')
# -
cv_df.mean()



