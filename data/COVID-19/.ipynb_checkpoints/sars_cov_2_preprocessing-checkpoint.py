# +
import pandas as pd
import os

#Read the viral protein accession number, sequence information
viral_info = pd.read_csv("sars_cov_2.fasta",header='infer',sep=",")
print(viral_info)

#Read the compound information (inchikey, name and canonical smiles)
compound_info = pd.read_csv("all_verified_keys.list",header='infer',sep=",")
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
interaction_df.to_csv("sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning_full_metadata.csv",index=False)

#Part of the interactions to be used for actual prediction
part_interaction_df = interaction_df.iloc[:,[0,3,4,6,7]]
part_interaction_df.to_csv("../sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning.csv",index=False)
# -

#Combine train, test and sars_cov_2 dataset to have information about all compounds and viral proteins for downstream torchtext prediction files
train_df = pd.read_csv("../Train_Compound_Viral_interactions_for_Supervised_Learning.csv",header='infer')
test_df = pd.read_csv("../Test_Compound_Viral_interactions_for_Supervised_Learning.csv",header='infer')
all_df = pd.concat([train_df,test_df,part_interaction_df],axis=0,ignore_index=True)
all_df.to_csv("../all_compound_viral_interactions_for_supervised_learning.csv",index=False)


