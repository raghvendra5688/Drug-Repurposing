#!/usr/bin/python -u
# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

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
#Parse the big data with latent space
big1_df = pd.read_csv("../data/Compound_Virus_Interactions/chembl_Filtered_Drug_Viral_proteins_Network_with_Drug_LS.csv",header='infer')
big2_df = pd.read_csv("../data/Drug_Protein_Networks/Thomas_Filtered_Drug_Viral_proteins_Network_with_Drug_LS.csv",header='infer')

#Write interactions with protein id, protein sequence, drug inchi key, drug smiles, pchembl value 
interaction1_df = big1_df.iloc[:,[0,4,5,8,7]].copy()
interaction2_df = big2_df.iloc[:,[0,4,5,8,7]].copy()

print(interaction1_df.columns)
print(interaction2_df.columns)

interaction1_df = refine_protein_sequences(interaction1_df,2000)
interaction2_df = refine_protein_sequences(interaction2_df,2000)

#Write the interaction data frame with the revisions
interaction1_df.to_csv("../data/Drug_Protein_Networks/Filtered_Drug_Viral_interactions_for_Supervised_Learning.csv",index=False)
interaction2_df.to_csv("../data/Drug_Protein_Networks/Thomas_Filtered_Drug_Viral_interactions_for_Supervised_Learning.csv",index=False)

interaction_df = pd.concat([interaction1_df,interaction2_df],ignore_index=True)
interaction_df.drop_duplicates(subset=['uniprot_accession','standard_inchi_key'],inplace=True)
interaction_df.reset_index(inplace=True, drop=True) 
interaction_df.to_csv("../data/Drug_Protein_Networks/Combined_Drug_Viral_interactions_for_Supervised_Learning.csv",index=False)
interaction_df

# +
#Parse the big data with latent space
total_length = len(big1_df.columns)
interaction1_df = big1_df.iloc[:,np.r_[0,4,5,8,7,12:total_length]].copy()
interaction2_df = big2_df.iloc[:,np.r_[0,4,5,8,7,12:total_length]].copy()

interaction1_df = refine_protein_sequences(interaction1_df,2000)
interaction2_df = refine_protein_sequences(interaction2_df,2000)

interaction_df = pd.concat([interaction1_df,interaction2_df],ignore_index=True)
interaction_df.drop_duplicates(subset=['uniprot_accession','standard_inchi_key'],inplace=True)
interaction_df.reset_index(inplace=True, drop=True) 
interaction_df.to_csv("../data/Drug_Protein_Networks/Combined_Drug_Viral_interactions_with_LS_for_Supervised_Learning.csv",index=False)
interaction_df
# -

np.size(np.union1d(big1_df['organism'].unique(),big2_df['organism'].unique()))

# +
#Create the train test split to be used by all downstream ML methods
y = interaction_df["pchembl_value"].values
indices = np.arange(interaction_df.shape[0])

_,_,_,_, indices_train, indices_test = train_test_split(interaction_df, y, indices, test_size=0.1, random_state=42)

indices_train,indices_test = list(indices_train),list(indices_test)
indices_list = []
for i in range(len(indices_train)):
    indices_list.append(['train',indices_train[i]])
for i in range(len(indices_test)):
    indices_list.append(['test',indices_test[i]])

indices_df = pd.DataFrame(indices_list, columns=['split','ids'])
indices_df
indices_df.to_csv("../data/Drug_Protein_Networks/Combined_Train_Test_Split_Info.csv",index=False)
print(len(indices_train),len(indices_test))


# +
#Split the big drug virus sequence data into train and test
big_df = pd.read_csv("../data/Drug_Protein_Networks/Combined_Drug_Viral_interactions_for_Supervised_Learning.csv",header='infer',sep=",")
indices_df = pd.read_csv("../data/Drug_Protein_Networks/Combined_Train_Test_Split_Info.csv",header='infer',sep=',')

indices_train = indices_df.loc[indices_df['split']=='train','ids'].values.tolist()
indices_test = indices_df.loc[indices_df['split']=='test','ids'].values.tolist()
train_df,test_df = train_test_set(big_df, indices_train, indices_test)

train_df.to_csv("./data/Train_Drug_Viral_interactions_for_Supervised_Learning.csv",index=False)
test_df.to_csv("./data/Test_Drug_Viral_interactions_for_Supervised_Learning.csv",index=False)

#Split the big drug virus LS data into train and test
big2_df = pd.read_csv("../data/Drug_Protein_Networks/Combined_Drug_Viral_interactions_with_LS_for_Supervised_Learning.csv",header='infer',sep=",")
train2_df, test2_df = train_test_set(big2_df,indices_train,indices_test)
#train2_df.to_csv("./data/Train_Drug_Viral_interactions_with_LS_for_Supervised_Learning.csv",index=False)
#test2_df.to_csv("./data/Test_Drug_Viral_interactions_with_LS_for_Supervised_Learning.csv",index=False)
train2_df
# -
train_sequences = train2_df["Sequence"]
train_sequences.to_csv("../CNN_AUTOENCODER/encoder/train_sequences",index=False)
test_sequences = test2_df["Sequence"]
test_sequences.to_csv("../CNN_AUTOENCODER/encoder/test_sequences",index=False)

# +
train_sequences_ls = pd.read_csv("../CNN_AUTOENCODER/encoder/train_sequences_lat.csv",header=None,sep=",")
test_sequences_ls = pd.read_csv("../CNN_AUTOENCODER/encoder/test_sequences_lat.csv",header=None,sep=",")
train2_df.reset_index(inplace=True,drop=True)
test2_df.reset_index(inplace=True,drop=True)

train_final_with_ls = pd.concat([train2_df,train_sequences_ls],axis=1)
test_final_with_ls = pd.concat([test2_df,test_sequences_ls],axis=1)

protein_ls_dic = {}
for i in range(64):
    protein_ls_dic[i] = 'P_LS_'+str(i)

train_final_with_ls.rename(columns=protein_ls_dic,inplace=True)
test_final_with_ls.rename(columns=protein_ls_dic,inplace=True)
# -


train_final_with_ls.to_csv("../data/Data_Supervised_Learning/Train_Drug_Viral_interactions_with_LS_v2_for_Supervised_Learning.csv",index=False)
test_final_with_ls.to_csv("../data/Data_Supervised_Learning/Test_Drug_Viral_interactions_with_LS_v2_for_Supervised_Learning.csv",index=False)


