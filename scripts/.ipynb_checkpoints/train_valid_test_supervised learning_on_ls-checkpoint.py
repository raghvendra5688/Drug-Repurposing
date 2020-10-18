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
#Load the train and test files with pchembl values (used for end-to-end deep learning)
train_with_label_df = pd.read_csv("../data/Train_Compound_Viral_interactions_for_Supervised_Learning.csv",header='infer')
test_with_label_df = pd.read_csv("../data/Test_Compound_Viral_interactions_for_Supervised_Learning.csv",header='infer')

print(train_with_label_df.shape)
print(test_with_label_df.shape)

#Load the train and test embedding representation for compounds
train_compound_df = pd.read_csv("../data/Train_Compound_LS.csv",header='infer')
test_compound_df = pd.read_csv("../data/Test_Compound_LS.csv",header='infer')

print(train_compound_df.shape)
print(test_compound_df.shape)

#Load the train and test morgan fingerprint representation for compounds
train_compound_mfp_df = pd.read_csv("../data/Train_Compound_MFP.csv",header='infer')
test_compound_mfp_df = pd.read_csv("../data/Test_Compound_MFP.csv",header='infer')

print(train_compound_mfp_df.shape)
print(test_compound_mfp_df.shape)

#Load the train and test embedding representation for proteins
train_protein_df = pd.read_csv("../data/Train_Protein_LS.csv",header=None)
test_protein_df = pd.read_csv("../data/Test_Protein_LS.csv",header=None)

column_names = ['PLS_'+str(i) for i in range(64)]
train_protein_df.columns = column_names
test_protein_df.columns = column_names


# -

def combine_df(df1,df2,df3):
    out_df = pd.concat([df1,df2,df3],axis=1,ignore_index=True)
    out_df.columns = df1.columns.tolist() + df2.columns.tolist() + df3.columns.tolist()
    return(out_df)


# +
#Obtain training and testing data frame with ls for compounds and ls for proteins
train_combined_ls_df = combine_df(train_with_label_df,train_compound_df, train_protein_df)
test_combined_ls_df = combine_df(test_with_label_df, test_compound_df, test_protein_df)

train_combined_ls_df.to_csv("../data/Train_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv",index=False)
test_combined_ls_df.to_csv("../data/Test_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv",index=False)


#Obtain training and testing data frame with morgan fingerprints for compounds and ls for proteins
train_combined_mfp_ls_df = combine_df(train_with_label_df,train_compound_mfp_df, train_protein_df)
test_combined_mfp_ls_df = combine_df(test_with_label_df, test_compound_mfp_df, test_protein_df)

train_combined_mfp_ls_df.to_csv("../data/Train_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv",index=False)
test_combined_mfp_ls_df.to_csv("../data/Test_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv",index=False)
# -

