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

import pubchempy as pcp
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
# +
#Take the union of compound names from all the sources (Barabasi paper, Drug Virus info site, 12k compound paper)
compound_files_valid = pd.read_csv("compounds_barabasi.list",header=None)
compound_names_valid = compound_files_valid.iloc[:,0].values.tolist()

compound_virus_info = pd.read_csv("compound_virus_info.list",header=None)
compound_virus_names = compound_virus_info.iloc[:,1].values.tolist()

compound_12k_info = pd.read_csv("compounds_12k.csv",header='infer')
compound_12k_names = compound_12k_info["Compound_Name"].values.tolist()

all_compounds_to_test = list(set().union(compound_names_valid,compound_virus_names,compound_12k_names))
print(len(all_compounds_to_test))
# -

#Use pubchempy to get the pubchme id of all compounds in the list
compound_lists,counter = [],0
for compound_name in all_drugs_to_test:
    temp_compound_info = pcp.get_compounds(compound_name, 'name')
    print(counter)
    if (temp_compound_info!=[]):
        compound_lists.append(temp_compound_info[0])
    counter=counter+1


print(len(compound_lists))

#Get the inchikey, canonical smiles and compound names for those compound which satisfy the constraints on SMILES lengths and don't contain salt bridges
standard_inchikey, canonical_smiles, compound_names = [],[]
for compound in compound_lists:
    temp_smiles = compound.isomeric_smiles
    temp_inchikey  = compound.inchikey
    temp_name = compound.name
    m = Chem.MolFromSmiles(temp_smiles)
    rev_smiles = Chem.MolToSmiles(m,isomericSmiles=False)
    if ('.' not in rev_smiles and len(rev_smiles)<=128 and len(rev_smiles)>=10):
        canonical_smiles.append(rev_smiles)
        standard_inchikey.append(temp_inchikey)
        compound_names.append(temp_name)

#Write the information about these compounds in output file
out_df = pd.DataFrame({'standard_inchi_key':standard_inchikey ,'compound_name':compound_names, 'canonical_smiles':canonical_smiles})
out_df.to_csv("../../Drug-Repurposing/data/COVID-19/all_verified_keys.list",index=False)


