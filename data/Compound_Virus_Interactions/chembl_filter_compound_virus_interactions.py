# ---
# jupyter:
#   jupytext:
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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import gzip

# +
#Get the compound - virus interaction network
compound_viral_protein_info = pd.read_csv("Compound_Viral_proteins_Network.csv",header='infer',sep="\t")


#Get rows with missing pChembl Values and remove them
bool_series = pd.notnull(compound_viral_protein_info['pchembl_value'])
print("Missing Chembl Items: ",sum(pd.isnull(compound_viral_protein_info['pchembl_value'])))
subset0_compound_viral_protein_info = compound_viral_protein_info[bool_series]
print("Shape initially: ",subset0_compound_viral_protein_info.shape)


#Select compound viral interactions with standard type IC50, Ki, Kd
subset1_compound_viral_protein_info = subset0_compound_viral_protein_info.loc[subset0_compound_viral_protein_info['standard_type'].isin(['Potency','IC50','Ki','Kd'])]
print("Shape after standard type selection: ",subset1_compound_viral_protein_info.shape)


#Fix the smiles to be in standard representation
smiles_info = subset1_compound_viral_protein_info[["standard_inchi_key","canonical_smiles"]].values.tolist()
smiles_info = list(set(tuple(x) for x in smiles_info))


canonical_smiles_list, inchikeys_smiles_list = [],[]
for i in range(len(smiles_info)):
    canonical_smiles_list.append(smiles_info[i][1])
    inchikeys_smiles_list.append(smiles_info[i][0])
    
#Remove compounds which have salt bridges and non-molecules
res1,res2 = [],[]
for i in range(len(canonical_smiles_list)):
    smiles = canonical_smiles_list[i]
    inchikey = inchikeys_smiles_list[i]
    m = Chem.MolFromSmiles(smiles)
    if m is not None:
        rev_smiles = Chem.MolToSmiles(m,isomericSmiles=False)
        #If molecule has salt bridge remove it
        if ('.' not in rev_smiles):
            res1.append(rev_smiles)
            res2.append(inchikey)


#Use only compounds whose length is less than 128
to_use_compound_list = [x for x in res1 if len(x)<=128 and len(x)>=10]
to_use_ids = [i for i in range(len(res1)) if len(res1[i])<=128 and len(res1[i])>=10]


#Write the compound list in form readable for LSTM autoencoder
compound_info = pd.DataFrame({'src':to_use_compound_list,'trg':to_use_compound_list})
compound_info.to_csv("chembl_compound_src_target_info.csv",index=False)


#Get the unique Inchi key id of all compounds with length <= 128
rev_inchikeys_smiles_list = [res2[i] for i in to_use_ids]
print("No of unique compounds: ",len(rev_inchikeys_smiles_list))


#Get compound viral interactions with compounds of length <= 128
subset2_compound_viral_protein_info = subset1_compound_viral_protein_info.loc[subset1_compound_viral_protein_info["standard_inchi_key"].isin(rev_inchikeys_smiles_list)]

#Remove repeated compound, viral protein combinations with multiple standard type (keep one)
subset2_compound_viral_protein_info = subset2_compound_viral_protein_info.drop_duplicates(subset=["uniprot_accession","standard_inchi_key"])
print("Shape after compound selection: ",subset2_compound_viral_protein_info.shape)
subset2_compound_viral_protein_info
# -

#Distribution of pchembl value based on standard type
subset2_compound_viral_protein_info["standard_type"].value_counts()

#Get info about unique compounds
only_compound_info = [[rev_inchikeys_smiles_list[i],to_use_compound_list[i]]for i in range(len(to_use_compound_list))]
only_compound_info = pd.DataFrame(only_compound_info, columns=['standard_inchi_key','canonical_smiles'])
only_compound_info.to_csv("chembl_Filtered_Compounds.csv",index=False)
only_compound_info.shape

# +
#Get info about unique viral accession numbers
only_viral_info = subset2_compound_viral_protein_info[["uniprot_accession","organism","target_pref_name"]].values.tolist()
only_viral_info = set(tuple(x) for x in only_viral_info)
only_viral_info = pd.DataFrame(only_viral_info,columns=["uniprot_accession","organism","target_pref_name"])
print("Unique Viral Information in Filtered data: ",only_viral_info.shape)

#Get viral sequence information about all virsuses in Uniprot and intersect with ChEMBL
full_viral_proteins = pd.read_csv("Full_Viral_proteins_with_Uniprot_IDs.csv.gz",compression='gzip')
full_viral_proteins.rename({"Uniprot_id":"uniprot_accession"},axis=1, inplace=True)
chembl_viral_proteins_with_sequences = pd.merge(only_viral_info,full_viral_proteins.iloc[:,[0,1,3]],on="uniprot_accession")

#Get info about unique set of viral proteins
chembl_viral_proteins_with_sequences.sort_values(by=["uniprot_accession"])
chembl_viral_proteins_with_sequences.to_csv("chembl_Filtered_Viral_Proteins.csv",index=False)
seq_lens = [len(x) for x in chembl_viral_proteins_with_sequences["Sequence"].values.tolist()]
chembl_viral_proteins_with_sequences

# +
#Combine the compound viral interactions with protein sequence information
subset3_compound_viral_protein_info = pd.merge( chembl_viral_proteins_with_sequences,
                                            subset2_compound_viral_protein_info.iloc[:,[0,1,2,5,6,7]], 
                                            on=['uniprot_accession','organism','target_pref_name'])

subset3_compound_viral_protein_info = subset3_compound_viral_protein_info.drop_duplicates(subset=['uniprot_accession','standard_inchi_key'])

print("No of organisms: ",len(subset3_compound_viral_protein_info['organism'].unique()))
print("No of viral proteins: ",len(subset3_compound_viral_protein_info['uniprot_accession'].unique()))
print("No of compounds: ",len(subset3_compound_viral_protein_info['standard_inchi_key'].unique()))
print("Measurement types:\n",subset3_compound_viral_protein_info['standard_type'].value_counts())
print(subset3_compound_viral_protein_info.shape)
subset3_compound_viral_protein_info.to_csv("chembl_Filtered_Compound_Viral_proteins_Network.csv",index=False)
subset3_compound_viral_protein_info
# -


