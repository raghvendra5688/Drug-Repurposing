#!/usr/bin/python -u
# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import argparse

SEED = 123
random.seed(SEED)
np.random.seed(SEED)


def combine_df(df1,df2,df3):
    out_df = pd.concat([df1,df2,df3],axis=1,ignore_index=True)
    out_df.columns = df1.columns.tolist() + df2.columns.tolist() + df3.columns.tolist()
    return(out_df)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Create the test file for sars-cov-2 compound-viral protein interactions with embedding for compounds + embeddings for proteins')
    parser.add_argument('input1', help='The sars-cov-2 compound-viral interactions file for end-to-end deep learning')
    parser.add_argument('input2', help='File containging autoencoder embedding for compounds to be tested')
    parser.add_argument('input3', help='File containging fingerprints for compounds to be tested')
    parser.add_argument('input4', help='File containging autoencoder for sars-cov-2 vrial proteins to be tested')
    parser.add_argument('output1', help='Compound-viral protein embedding combining autoencoder embedding of compounds with autoencoder embedding of proteins')
    parser.add_argument('output2', help='Compound-viral protein embedding combining morgan fingerprints for compounds with autoencoder embedding of proteins')
    args = parser.parse_args()

    # +
    #Load the test files with pchembl values (used for end-to-end deep learning)
    test_with_label_df = pd.read_csv("../data/"+args.input1,header='infer')
    print("Loaded all interactions")

    #Load the test embedding representation for compounds
    test_compound_df = pd.read_csv("../data/"+args.input2,header='infer')

    print("Loaded the autoencoder embedding")

    #Load the test morgan fingerprint representation for compounds
    test_compound_mfp_df = pd.read_csv("../data/"+args.input3,header='infer')

    print("Loaded the fingerprint embedding")

    #Load the test embedding representation for proteins
    test_protein_df = pd.read_csv("../data/"+args.input4,header=None)
    print("Loaded the protein embedding")

    column_names = ['PLS_'+str(i) for i in range(64)]
    test_protein_df.columns = column_names

    # -

    # +
    #Obtain training and testing data frame with ls for compounds and ls for proteins
    test_combined_ls_df = combine_df(test_with_label_df, test_compound_df, test_protein_df)

    test_combined_ls_df.to_csv("../data/"+args.output1,index=False)
    print("SMILES Embedding + Protein Embedding file generated")

    #Obtain training and testing data frame with morgan fingerprints for compounds and ls for proteins
    test_combined_mfp_ls_df = combine_df(test_with_label_df, test_compound_mfp_df, test_protein_df)

    test_combined_mfp_ls_df.to_csv("../data/"+args.output2,index=False)
    print("Morgan Fingerprint + Protein Embedding file generated")
    # -
