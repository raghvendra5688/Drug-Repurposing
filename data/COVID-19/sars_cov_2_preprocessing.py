# +
import pandas as pd
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare the compound-viral activity test file for Sars-Cov-2 viral proteins')
    parser.add_argument('input1', help='Fasta file containing information about Sars-Cov-2 viral proteins')
    parser.add_argument('input2', help='File containing standard_inchi_key, names and canonical smiles of compounds of interest obtained as result of get_all_compounds_COVID_19.py')
    parser.add_argument('output', help='Test compound-viral protein interactions file for Sars-Cov-2 virus')
    args = parser.parse_args()


    #Read the viral protein accession number, sequence information
    viral_info = pd.read_csv(args.input1,header='infer',sep=",")
    print(viral_info)

    #Read the compound information (inchikey, name and canonical smiles)
    compound_info = pd.read_csv(args.input2,header='infer',sep=",")
    compound_info.columns=['standard_inchi_key',"compound_name","canonical_smiles"]
    print(compound_info)

    #Remove duplicate compounds
    #compound_info.drop_duplicates(subset=['standard_inchi_key'],inplace=True,keep='first')
    #compound_info.to_csv("all_verified_keys.list",index=False)

    # +
    #Make combinations of viral protein and compounds for testing
    viral_df = viral_info.loc[viral_info.index.repeat(compound_info.shape[0])].reset_index(drop=True)
    compound_df = pd.concat([compound_info]*viral_info.shape[0],axis=0,ignore_index=True)
    interaction_df = pd.concat([viral_df,compound_df],axis=1, ignore_index=True)

    #Fix the pchembl_value to a dummy value of 0
    interaction_df['pchembl_value']=0.0

    #Fix the column names
    column_names = ["uniprot_accession","Protein_Fragment","organism","Sequence","standard_inchi_key","compound_name","canonical_smiles","pchembl_value"]
    interaction_df.columns = column_names

    #Write the interaction information with full metadata
    print("Writing Compound-Viral activity with full metadata")
    interaction_df.to_csv("sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning_full_metadata.csv",index=False)

    #Part of the interactions to be used for actual prediction
    part_interaction_df = interaction_df.iloc[:,[0,3,4,6,7]]
    print("Writing Compound-Viral activity in format acceptable for supervised learning models")
    part_interaction_df.to_csv("../"+args.output,index=False)
    # -

    #Combine train, test and sars_cov_2 dataset to have information about all compounds and viral proteins for downstream torchtext prediction files
    train_df = pd.read_csv("../Train_Compound_Viral_interactions_for_Supervised_Learning.csv",header='infer')
    test_df = pd.read_csv("../Test_Compound_Viral_interactions_for_Supervised_Learning.csv",header='infer')
    all_df = pd.concat([train_df,test_df,part_interaction_df],axis=0,ignore_index=True)
    all_df.to_csv("../all_compound_viral_interactions_for_supervised_learning.csv",index=False)

    print("Generated the test file containing compound-viral protein interactions, whose activity values are to be predicted")
