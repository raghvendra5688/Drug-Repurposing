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

# ## 1. Dataset - Finding additional relevant compounds with activities

# Machine learning methods tend to work better the more data you have, and with this new virus popping up, data is a bit harder to come by. However, we do know a lot about how viruses work and relying on a main protease is common between many viruses. Available anti-viral compounds often target their proteases. 
#
# SARS and MERS are both coronavirus variants that are very similar and since their respective outbreaks, many biological assays have been done to test compounds on their main proteases. Bioactivities measured in papers by medicinal chemists and biochemists are tracked by NCBI and are freely available. A database of protease inhibitors will be built using this data.

# ### General Dataset Preparation

# Note that each of these needs to be run for each of the search results and gives a list of assay ID's, or AIDs.
#
# The searches used to generate a good AID list are:
#
# 1. Protein target GI73745819 - SARS Protease - Called SARS_C3_Assays.txt in this report
#
# 2. Protein target GI75593047 - HIV pol polyprotein - Called HIV_Protease_Assays.txt in this report
#
# 3. NS3 - Hep3 protease - Called NS3_Protease_Assays.txt in this report
#
# 4. 3CL-Pro - Mers Protease - Called MERS_Protease_Assays.txt in this report
#
# The actual compound activity data will be downloaded from NCBI using their system called "PUG-REST" which are specifically designed URLs that let you download raw info of various NCBI records.

#Imports
import rdkit
from rdkit.Chem import AllChem as Chem
from rdkit.DataStructs import cDataStructs
import numpy as np
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import os
import time
import pickle
import csv
from rdkit.Chem import QED
import random
import json
from sklearn.preprocessing import StandardScaler


# +
def get_assays(assay_path, assay_pickle_path):
    with open(str(assay_path)) as f:
        r = csv.reader(f)
        AIDs = list(r)
    assays = []
    for i, AID in zip(range(len(AIDs)), AIDs):
        #This needs to be changed to 
        #os.system('curl https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/sdf -o cmp.sdf' %CID)
        #if you run it on a mac
        os.system(f'wget https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{str(AID[0])}/csv -O additional_data/assay.csv')
        if os.stat(f'additional_data/assay.csv').st_size != 0:
            assays.append(pd.read_csv(f'additional_data/assay.csv'))

    pickle.dump(assays, open(str(assay_pickle_path), "wb"))

def get_mols_for_assays(assays_no_mol_path, assays_with_mol_path):
    assays = pickle.load(open(str(assays_no_mol_path), "rb"))
    for assay in assays:
        if len(assay) != 1:
            cids = list(assay[['PUBCHEM_CID']].values.astype("int32").squeeze())
            nan_counter = 0
            for i in range(len(cids)):
                if cids[i] < 0:
                    nan_counter += 1
                else:
                    break
            cids = cids[nan_counter:]
            mols = []
            for CID in cids:
                #os.system('curl https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/sdf -o cmp.sdf' %CID)
                os.system('wget https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/sdf -O additional_data/cmp.sdf' %CID)
                if os.stat(f'additional_data/cmp.sdf').st_size != 0:
                    mols.append(Chem.SDMolSupplier("additional_data/cmp.sdf")[0])
                else:
                    mols.append(None)

            for i in range(nan_counter):
                mols.insert(0,None)

            assay.insert(3, "Mol Object", mols)

    pickle.dump(assays, open(str(assays_with_mol_path), "wb"))


# +
#Get the information through the PUG-REST API which is relatively slow for each of the 4 viruses similar to SARS-COV-2

#get_assays("additional_data/SARS_C3_Assays_AID_only.csv", "additional_data/sars_assays_no_mol.pkl")
#get_mols_for_assays("additional_data/sars_assays_no_mol.pkl", "additional_data/sars_assays.pkl")

# +
#This goes an HTTP get on EVERY compound and takes a WHILE. Might be better to just use the pickled datasets
#get_assays("additional_data/MERS_Protease_Assays_AID_only.csv", "additional_data/mers_assays_no_mol.pkl")
#get_mols_for_assays("additional_data/mers_assays_no_mol.pkl", "additional_data/mers_assays.pkl")
#get_assays("additional_data/NS3_Protease_Assays_AID_only.csv", "additional_data/ns3_assays_no_mol.pkl")
#get_mols_for_assays("additional_data/ns3_assays_no_mol.pkl", "additional_data/ns3_assays.pkl")
#get_assays("additional_data/HIV_Protease_Assays_AID_only.csv", "additional_data/iv_assays_no_mol.pkl")
#get_mols_for_assays("additional_data/hiv_assays_no_mol.pkl", "additional_data/hiv_assays.pkl")
# -

#This datastructure is a dictionary of lists of dataframe already prepared
assays = {}
assays["sars"] = pickle.load(open("additional_data/sars_assays.pkl", "rb"))
assays["mers"] = pickle.load(open("additional_data/mers_assays.pkl", "rb"))
assays["ns3"] = pickle.load(open("additional_data/ns3_assays.pkl", "rb"))
assays["hiv"] = pickle.load(open("additional_data/hiv_assays.pkl", "rb"))
#os.system('rm additional_data/*.pkl additional_data/*.sdf additional_data/assay.csv')

# It is worth mentioning here the different kinds of Bioactivities that an assay can report. Depending on what was relevant to the scientists involved in the study, various values can be used. Possibly most importantly for generating this dataset though is to not confuse the different kinds of activities. We will focus on IC50, which is the concentration of the compound at which 50% inhibition is observed. The value is normal reported as a "Micromolar concentration". The lower the value, the better the compound is at inhibiting the protein. It is important to not be tempted to use the "Activity" reported in some assays, which is normally a % and corresponds to how much that compound inhibits the protein at a given concentration. We're sticking with IC50 because this value is very information rich and actually many "Activity" experiments go into producing 1 IC50 value. Also they are more easily comparable, as we don't need to standardize concentration across the assays.

# For this report we will focus on the "PubChem Standard Value" which is normally a standardized value using some metric (we will further narrow to only the metrics we want)

#This removes all the assays that do not have a column called "PubChem Standard Value"
for a in ["sars", "mers", "ns3", "hiv"]:
    print("Length of",str(a),"before removing")
    print(len(assays[a]))
    assays[a] = np.array(assays[a])
    bad_list = []
    good_list = []
    for i in range(len(assays[a])):
        ic50_cols = [col for col in assays[a][i].columns if 'PubChem Standard Value' in col]
        if not ic50_cols:
            bad_list.append(i)
        else:
            good_list.append(int(i))

    bad_list = np.array(bad_list)
    good_list = np.array(good_list, dtype='int32')

    assays[a] = assays[a][good_list]
    print("Length of",str(a),"after removing")
    print(len(assays[a]))

#Remove unnesessary columns
a1 = {'sars':'SARS coronavirus',
      'mers':'Middle East respiratory syndrome-related coronavirus',
      'ns3':'Hepacivirus C',
      'hiv':'Human immunodeficiency virus 1'}
for a in ["sars", "mers", "ns3", "hiv"]:
    for i in range(len(assays[a])):
        assays[a][i] = assays[a][i][["Mol Object", "PubChem Standard Value", "Standard Type"]]
        assays[a][i]['organism'] = a1[a]

#### Look at what different kind of metrics were used
for a in ["sars", "mers", "ns3", "hiv"]:
    for i in range(len(assays[a])):
        print(assays[a][i][["Standard Type"]].values[-1])

# You can see that even the "standard" values can have quite a variance in what they mean. As mentioned above, we will focus on only IC50 values. We know from enzyme kinetics that when a ligand binds to a protein in an uncompetetive scenario (i.e. an assay) the Ki value determined is equal to the IC50, so we can include it too. Also the Kd value is a more general way of referring to the Ki value, so it can be included. 

#concatenate all of the dataframe in the dictionary into a single list.
#We lose the notion that they were once for different targets
all_dfs = []
for a in ["sars", "mers", "ns3", "hiv"]:
    for i in range(len(assays[a])):
        if assays[a][i][["Standard Type"]].values[-1][0] in {"IC50", "Ki", "Kd"}:
            print(assays[a][i])
            all_dfs.append(assays[a][i])

#Remove header info and concatenate them
for i in range(len(all_dfs)):
    all_dfs[i] = all_dfs[i].iloc[4:]
final_df = pd.concat(all_dfs)

#Take all the compounds with activites converted to -log10(x nM)
final_df['PubChem Standard Value'] = final_df['PubChem Standard Value'].astype(float)
final_df = final_df[final_df['PubChem Standard Value'].notna()]
final_df['pchembl_value'] = -np.log10(final_df['PubChem Standard Value'])+6
final_df

pickle.dump(final_df, open("additional_data/final_df.pkl", "wb"))

# ### Method Specific-preparation

# Now moving on to preparing the dataset for use in the predictive model 

df = pickle.load(open("additional_data/final_df.pkl", "rb"))
df.index = range(df.shape[0])

#Remove samples which are not molecules
ids = []
all_molecules = df[['Mol Object']].values[:,0]
for i in range(df.shape[0]):
    mol = all_molecules[i]
    if not mol:
        ids.append(i)
rev_df = df.drop(df.index[ids])
rev_df.index

rev_df.insert(5, 'canonical_smiles', [Chem.MolToSmiles(x, isomericSmiles=False) for x in rev_df[['Mol Object']].values[:,0]], True)
rev_df.insert(6, 'standard_inchi_key', [Chem.inchi.MolToInchiKey(x) for x in rev_df[['Mol Object']].values[:,0]], True)
rev_df

#Remove samples where compounds have SMILES with length>128 or length<10
salt_indexes = []
rev_df = rev_df.reset_index()
for i in range(len(rev_df)):
    if "." in rev_df[["canonical_smiles"]].values[i][0]:
        salt_indexes.append(i)
    if len(rev_df[['canonical_smiles']].values[i][0])>128 or len(rev_df[['canonical_smiles']].values[i][0])<10:
        salt_indexes.append(i)

rev_df = rev_df.drop(rev_df.index[salt_indexes])
load_sequence_info = pd.read_csv("ncbi_Filtered_Viral_Proteins.csv",header='infer',sep=",")
load_sequence_info

only_drug_info = rev_df[["standard_inchi_key","canonical_smiles"]].values.tolist()
only_drug_info = set(tuple(x) for x in only_drug_info)
only_drug_info = pd.DataFrame(only_drug_info, columns=['standard_inchi_key','canonical_smiles'])
only_drug_info.to_csv("ncbi_Filtered_Compounds.csv",index=False)
only_drug_info

rev_df.rename({"Standard Type":"standard_type"},axis=1,inplace=True)
output_df = pd.merge(load_sequence_info,rev_df.iloc[:,[4,7,6,3,5]],on="organism",how="right",sort=True)
output_df.to_csv("ncbi_Filtered_Compound_Viral_proteins_Network.csv",index=False)
output_df.drop_duplicates(subset=["uniprot_accession"])

# +
drug_list = only_drug_info["canonical_smiles"].values.tolist()

#Write the drug list in form readable for LSTM autoencoder
drug_info = pd.DataFrame({'src':drug_list,'trg':drug_list})
drug_info.to_csv("ncbi_compound_src_target_info.csv",index=False)
drug_list
# -

