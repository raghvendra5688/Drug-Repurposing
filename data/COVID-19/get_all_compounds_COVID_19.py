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
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

# +

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Provide list of compounds from barabasi paper, drug virus info and ReframeDB')
    parser.add_argument('input1', help='List of compounds from Barabasi paper')
    parser.add_argument('input2', help='List of compounds from drugvirus.info')
    parser.add_argument('input3', help='List of compounds from ReframeDB')
    parser.add_argument('output', help='Output list of compound standard_inchi_key, name, canonical smiles')
    args = parser.parse_args()

    #Take the union of compound names from all the sources (Barabasi paper, Drug Virus info site, 12k compound paper)
    compound_files_valid = pd.read_csv(args.input1,header=None)
    compound_names_valid = compound_files_valid.iloc[:,0].values.tolist()

    compound_virus_info = pd.read_csv(args.input2,header=None)
    compound_virus_names = compound_virus_info.iloc[:,1].values.tolist()

    compound_12k_info = pd.read_csv(args.input3,header='infer')
    compound_12k_names = compound_12k_info["Compound_Name"].values.tolist()

    all_compounds_to_test = list(set().union(compound_names_valid,compound_virus_names,compound_12k_names))
    print(len(all_compounds_to_test))
    # -

    #Use pubchempy to get the pubchme id of all compounds in the list
    print("Downloading compound info from PubChem using PUG-REST API")
    compound_lists,compound_names,counter = [],[],0
    for compound_name in all_compounds_to_test:
	    temp_compound_info = pcp.get_compounds(compound_name, 'name')
	    print("Extracted compound no: ",counter)
	    if (temp_compound_info!=[]):
	        compound_lists.append(temp_compound_info[0])
	        compound_names.append(compound_name)
	    counter=counter+1


    print(len(compound_lists))

    #Get the inchikey, canonical smiles and compound names for those compound which satisfy the constraints on SMILES lengths and don't contain salt bridges
    standard_inchikey, canonical_smiles, final_compound_names, counter = [],[], [], 0
    for compound in compound_lists:
        temp_smiles = compound.isomeric_smiles
        temp_inchikey  = compound.inchikey
        temp_name = compound_names[counter]
        m = Chem.MolFromSmiles(temp_smiles)
        rev_smiles = Chem.MolToSmiles(m,isomericSmiles=False)
        if ('.' not in rev_smiles and len(rev_smiles)<=128 and len(rev_smiles)>=10):
            canonical_smiles.append(rev_smiles)
            standard_inchikey.append(temp_inchikey)
            final_compound_names.append(temp_name)
        counter = counter+1

    #Write the information about these compounds in output file
    out_df = pd.DataFrame({'standard_inchi_key':standard_inchikey ,'compound_name':final_compound_names, 'canonical_smiles':canonical_smiles})
    out_df.drop_duplicates(subset=['standard_inchi_key'],inplace=True,keep='first')
    out_df.to_csv(args.output,index=False)
    print("Finished downloading repurposable compound information from PubChem for SARS-COV-2 viral proteins")
