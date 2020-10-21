#!/usr/bin/python -u
# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import argparse

SEED = 123
random.seed(SEED)
np.random.seed(SEED)


# -
def train_test_set(df,train_ids,test_ids):
    train_df = df.iloc[train_ids,:]
    test_df = df.iloc[test_ids,:]
    return(train_df,test_df)


def refine_protein_sequences(df,cutoff):
    all_protein_sequences = df["Sequence"].unique()
    all_protein_sequences = sorted(all_protein_sequences,key=len)
    revised_protein_sequences={}
    for x in all_protein_sequences:
        if (len(x)<=cutoff):
            revised_protein_sequences[x]=x
        else:
            revised_protein_sequences[x]=x[:2000]
    df["Sequence"].replace(revised_protein_sequences,inplace=True)
    return df


# +

def get_train_test_samples(input1,input2,output1,output2):

    #Parse the big data with latent space
    big1_df = pd.read_csv("../data/Compound_Virus_Interactions/"+input1,header='infer')
    big2_df = pd.read_csv("../data/Compound_Virus_Interactions/"+input2,header='infer')

    #Write interactions with protein id, protein sequence, drug inchi key, drug smiles, pchembl value 
    interaction1_df = big1_df.iloc[:,[0,4,5,6,8]].copy()
    interaction2_df = big2_df.iloc[:,[0,4,5,6,8]].copy()

    print(interaction1_df.columns)
    print(interaction2_df.columns)

    interaction1_df = refine_protein_sequences(interaction1_df,2000)
    interaction2_df = refine_protein_sequences(interaction2_df,2000)

    #Write the interaction data frame with the revisions
    #interaction1_df.to_csv("../data/Drug_Protein_Networks/Filtered_Drug_Viral_interactions_for_Supervised_Learning.csv",index=False)
    #interaction2_df.to_csv("../data/Drug_Protein_Networks/Thomas_Filtered_Drug_Viral_interactions_for_Supervised_Learning.csv",index=False)

    interaction_df = pd.concat([interaction1_df,interaction2_df],ignore_index=True)
    interaction_df.drop_duplicates(subset=['uniprot_accession','standard_inchi_key'],inplace=True)
    interaction_df.reset_index(inplace=True, drop=True) 
    interaction_df
    print(interaction1_df.shape)
    print(interaction2_df.shape)
    # -

    #Unique no of viral organisms in the dataset
    print(np.size(np.union1d(big1_df['organism'].unique(),big2_df['organism'].unique())))
    plt.hist(interaction_df["pchembl_value"])

    # +
    #Create the train test split to be used by all downstream ML methods
    y = interaction_df["pchembl_value"].values
    indices = np.arange(interaction_df.shape[0])

    _,_,_,_, indices_train, indices_test = train_test_split(interaction_df, y, indices, test_size=0.1, random_state=42)

    indices_train,indices_test = list(indices_train),list(indices_test)
    indices_train_set = set(indices_train)
    indices_test_set = set(indices_test)

    indices_list = []
    for i in range(len(indices_train)):
        indices_list.append(['train',indices_train[i]])
    for i in range(len(indices_test)):
        indices_list.append(['test',indices_test[i]])

    indices_df = pd.DataFrame(indices_list, columns=['split','ids'])
    indices_df
    print(len(indices_train),len(indices_test))
    print(indices_df)


    # +
    #Split the big drug virus sequence data into train and test
    indices_train = indices_df.loc[indices_df['split']=='train','ids'].values.tolist()
    indices_test = indices_df.loc[indices_df['split']=='test','ids'].values.tolist()
    train_df,test_df = train_test_set(interaction_df, indices_train, indices_test)

    train_df.to_csv("../data/"+output1,index=False)
    test_df.to_csv("../data/"+output2,index=False)
    # -

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Divide compound-viral protein activities into training and test set for end-to-end deep learning')
    parser.add_argument('input1', help='Filtered compound-viral protein activities from ChEMBL')
    parser.add_argument('input2', help='Filtered compound-viral protein activities from NCBI')
    parser.add_argument('output_train', help='Train compound-viral protein interactions file')
    parser.add_argument('output_test', help='Test compound-viral protein interactions file')
    args = parser.parse_args()

    get_train_test_samples(args.input1,args.input2,args.output_train,args.output_test)
    print("Divided compound-viral protein activities into train and test set for deep learning")

