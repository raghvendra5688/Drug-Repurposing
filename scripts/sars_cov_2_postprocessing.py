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
import rankaggregation as ra

#Get list of all compound-sars-cov-2 viral protein interactions 
compound_viral_df = pd.read_csv("../data/COVID-19/sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning_full_metadata.csv",header='infer')
print("Loaded compound viral protein interactions for SARS-COV-2 viral proteins")
print(compound_viral_df.shape)


#For a given viral protein get ranked list of drugs for a particular ML method
def get_ranked_list(df,proteins,rev_drug_info,protein_mapping_dict,ranked_list_proteins):
    for i in range(len(proteins)):
        
        #Subset to single sars-cov-2 viral protein
        temp_df = df[df["uniprot_accession"]==proteins[i]].copy()
        
        #Order by predictions
        temp_df = temp_df.sort_values(by="predictions",ascending=False)
        
        #Subset to the same single sars-cov-2 viral protein
        temp_rev_drug_info = rev_drug_info[rev_drug_info["uniprot_accession"]==proteins[i]].copy()
        
        #Merge the two data frames to get compound names
        temp2_df = pd.merge(temp_df,temp_rev_drug_info,on=["uniprot_accession","standard_inchi_key"],how='left')
        temp2_df.drop_duplicates(inplace=True)
        
        temp2_df = temp2_df.sort_values(by="predictions",ascending=False)
        
        drug_info = temp2_df["compound_name"].values.tolist()
        ranked_list_proteins[protein_mapping_dict[proteins[i]]].append(drug_info)
    return(ranked_list_proteins)


#Aggregate the ranked list of drugs to get final set of ordered list of drugs
def per_protein_rank(ranked_list_proteins, protein_name):
    temp_list = ranked_list_proteins[protein_name]
    agg = ra.RankAggregator()
    return(agg.average_rank(temp_list))


# +
#Use compound_viral_df and results from ML methods to generate ranked list
rf_smiles_predictions = pd.read_csv("../results/rf_LS_Compound_LS_Protein_supervised_sars_cov_2_predictions.csv",header='infer',sep=",")
svm_smiles_predictions = pd.read_csv("../results/svm_LS_Compound_LS_Protein_supervised_sars_cov_2_predictions.csv",header='infer',sep=",")
xgb_smiles_predictions = pd.read_csv("../results/xgb_LS_Compound_LS_Protein_supervised_sars_cov_2_predictions.csv",header='infer',sep=",")

rf_mfp_predictions = pd.read_csv("../results/rf_MFP_Compound_LS_Protein_supervised_sars_cov_2_predictions.csv",header='infer',sep=",")
svm_mfp_predictions = pd.read_csv("../results/svm_MFP_Compound_LS_Protein_supervised_sars_cov_2_predictions.csv",header='infer',sep=",")
xgb_mfp_predictions = pd.read_csv("../results/xgb_MFP_Compound_LS_Protein_supervised_sars_cov_2_predictions.csv",header='infer',sep=",")

cnn_predictions = pd.read_csv("../results/cnn_supervised_sars_cov_2_predictions.csv",header='infer',sep=",")
lstm_predictions = pd.read_csv("../results/lstm_supervised_sars_cov_2_predictions.csv",header='infer',sep=",")
cnn_lstm_predictions = pd.read_csv("../results/cnn_lstm_supervised_sars_cov_2_predictions.csv",header='infer',sep=",")
gat_cnn_predictions = pd.read_csv("../results/gat_cnn_supervised_sars_cov_2_predictions.csv",header='infer',sep=',')

#Get a list of the unique proteins
all_proteins = rf_smiles_predictions["uniprot_accession"].unique()

#Create a dictionary of ranked list based on the 3 protein names
ranked_list_proteins = {}
protein_mapping_dict = {}
for i in range(len(all_proteins)):
    protein_fragment=compound_viral_df[compound_viral_df["uniprot_accession"]==all_proteins[i]]["Protein_Fragment"].unique()
    protein_fragment=protein_fragment[0]
    protein_mapping_dict[all_proteins[i]]=protein_fragment
    ranked_list_proteins[protein_fragment]=[]
    

#Get ranked list for each protein using ML methods except GLM
#ranked_list_proteins = get_ranked_list(rf_smiles_predictions, all_proteins, compound_viral_df, protein_mapping_dict, ranked_list_proteins)
#ranked_list_proteins = get_ranked_list(svm_smiles_predictions,all_proteins,compound_viral_df,protein_mapping_dict,ranked_list_proteins)
ranked_list_proteins = get_ranked_list(xgb_smiles_predictions,all_proteins,compound_viral_df,protein_mapping_dict,ranked_list_proteins)

#ranked_list_proteins = get_ranked_list(rf_mfp_predictions,all_proteins,compound_viral_df, protein_mapping_dict, ranked_list_proteins)
ranked_list_proteins = get_ranked_list(svm_mfp_predictions,all_proteins,compound_viral_df, protein_mapping_dict, ranked_list_proteins)
ranked_list_proteins = get_ranked_list(xgb_mfp_predictions,all_proteins,compound_viral_df, protein_mapping_dict, ranked_list_proteins)

ranked_list_proteins = get_ranked_list(cnn_predictions,all_proteins,compound_viral_df, protein_mapping_dict, ranked_list_proteins)
#ranked_list_proteins = get_ranked_list(lstm_predictions,all_proteins, compound_viral_df,protein_mapping_dict,ranked_list_proteins)
#ranked_list_proteins = get_ranked_list(cnn_lstm_predictions,all_proteins, compound_viral_df, protein_mapping_dict,ranked_list_proteins)
ranked_list_proteins = get_ranked_list(gat_cnn_predictions,all_proteins, compound_viral_df, protein_mapping_dict,ranked_list_proteins)

# +
##Perform rank aggregation per protein: this ranking strategy is not used
#protein_names=[]
#for i in range(len(all_proteins)):
#    protein_names.append(protein_mapping_dict[all_proteins[i]])
#print(protein_names)

##Get ranked list for each viral protein
#rankings = per_protein_rank(ranked_list_proteins,protein_names[0])
#rankings_df = pd.DataFrame(rankings,columns=['Drug','Overall Weight'])
#rankings_df['Protein_Fragment']=protein_names[0]
#rankings_df


# -

#Combine predictions to get rankings based on average predictions
def combined_df(df1,df2,df3,df4,df5,protein_id):
    temp_df1=df1[df1["uniprot_accession"]==protein_id]
    temp_df1=temp_df1.sort_values(by="standard_inchi_key")
    temp_df1 = temp_df1.reset_index(drop=True)
    
    temp_df2=df2[df2["uniprot_accession"]==protein_id]
    temp_df2=temp_df2.sort_values(by="standard_inchi_key")
    temp_df2 = temp_df2.reset_index(drop=True)
    
    temp_df3=df3[df3["uniprot_accession"]==protein_id]
    temp_df3=temp_df3.sort_values(by="standard_inchi_key")
    temp_df3 = temp_df3.reset_index(drop=True)
    
    temp_df4=df4[df4["uniprot_accession"]==protein_id]
    temp_df4=temp_df4.sort_values(by="standard_inchi_key")
    temp_df4 = temp_df4.reset_index(drop=True)
    
    temp_df5=df5[df5["uniprot_accession"]==protein_id]
    temp_df5=temp_df5.sort_values(by="standard_inchi_key")
    temp_df5 = temp_df5.reset_index(drop=True)
        
    
    final_df=pd.concat([temp_df1.iloc[:,0:3],temp_df2.iloc[:,2],
                                     temp_df3.iloc[:,2],temp_df4.iloc[:,2],
                                     temp_df5.iloc[:,2]],axis=1,join='inner',ignore_index=True)
    return(final_df)


#Combine predictions of models and rank based on average predicted pChEMBL values
def get_results_with_pchembl(final_combined_df,rev_drug_info,protein_name):
    average_combined_df = final_combined_df.iloc[:,[0,1]].copy()
    average_combined_df.columns=["uniprot_accession","standard_inchi_key"]
    average_combined_df["avg_predictions"]=final_combined_df.iloc[:,[2,3,4,5,6]].mean(axis=1)
    final_output_df = pd.merge(average_combined_df,rev_drug_info.iloc[:,[4,5,6]],on='standard_inchi_key')
    final_output_df.drop_duplicates(inplace=True)
    final_output_df["protein_fragment"]=protein_name
    final_output_df = final_output_df.iloc[:,[0,1,4,3,5,2]]
    final_output_df = final_output_df.sort_values("avg_predictions",ascending=False)
    final_output_df = final_output_df.reset_index(drop=True)
    return(final_output_df)


# +
#For PL-PRO (nsp3)
final_combined_df = combined_df(xgb_smiles_predictions,svm_mfp_predictions,xgb_mfp_predictions,cnn_predictions,gat_cnn_predictions,all_proteins[0])
output1_df = get_results_with_pchembl(final_combined_df,compound_viral_df,protein_mapping_dict[all_proteins[0]])
output1_df.to_csv("../results/PL_Pro_Top_Ranked_Compounds.csv",index=False)
print("Ranked list of top compounds for PL-PRO written")

#For 3-CL Pro
final_combined_df = combined_df(xgb_smiles_predictions,svm_mfp_predictions,xgb_mfp_predictions,cnn_predictions,gat_cnn_predictions,all_proteins[1])
output2_df = get_results_with_pchembl(final_combined_df,compound_viral_df,protein_mapping_dict[all_proteins[1]])
output2_df.to_csv("../results/3CL_Pro_Top_Ranked_Compounds.csv",index=False)
print("Ranked list of top compounds for 3Cl-PRO written")

#For Spike Protein
final_combined_df = combined_df(xgb_smiles_predictions,svm_mfp_predictions,xgb_mfp_predictions,cnn_predictions,gat_cnn_predictions,all_proteins[2])
output3_df = get_results_with_pchembl(final_combined_df,compound_viral_df,protein_mapping_dict[all_proteins[2]])
output3_df.to_csv("../results/Spike_Pro_Top_Ranked_Compounds.csv",index=False)
print("Ranked list of top compounds for Spike-PRO written")
# +
top_intersection_compounds = set(output1_df["compound_name"].values.tolist()[0:100]).intersection(set(output2_df["compound_name"].values.tolist()[0:100]), set(output3_df["compound_name"].values.tolist()[0:100]))

top_intersection_compounds_df = output1_df.loc[output1_df['compound_name'].isin(list(top_intersection_compounds)),
                                               ["compound_name","standard_inchi_key","canonical_smiles"]]
top_intersection_compounds_df.to_csv("../results/Top_Intersection_Compounds.csv",index=False)
print("Writing the top commonly appearning compounds for the 3 viral proteins")
