#!/usr/bin/python -u
# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
# +
#Load the test files with pchembl values (used for end-to-end deep learning)
test_with_label_df = pd.read_csv("../data/sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning.csv",header='infer')

print(test_with_label_df.shape)

#Load the test embedding representation for compounds
test_compound_df = pd.read_csv("../data/sars_cov_2_Compound_LS.csv",header='infer')

print(test_compound_df.shape)

#Load the test morgan fingerprint representation for compounds
test_compound_mfp_df = pd.read_csv("../data/sars_cov_2_Compound_MFP.csv",header='infer')

print(test_compound_mfp_df.shape)

#Load the test embedding representation for proteins
test_protein_df = pd.read_csv("../data/sars_cov_2_Protein_LS.csv",header=None)

column_names = ['PLS_'+str(i) for i in range(64)]
test_protein_df.columns = column_names

# -

def combine_df(df1,df2,df3):
    out_df = pd.concat([df1,df2,df3],axis=1,ignore_index=True)
    out_df.columns = df1.columns.tolist() + df2.columns.tolist() + df3.columns.tolist()
    return(out_df)


# +
#Obtain training and testing data frame with ls for compounds and ls for proteins
test_combined_ls_df = combine_df(test_with_label_df, test_compound_df, test_protein_df)

test_combined_ls_df.to_csv("../data/sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv",index=False)


#Obtain training and testing data frame with morgan fingerprints for compounds and ls for proteins
test_combined_mfp_ls_df = combine_df(test_with_label_df, test_compound_mfp_df, test_protein_df)

test_combined_mfp_ls_df.to_csv("../data/sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv",index=False)
# -


