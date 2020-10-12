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
#import rankaggregation as ra

# +
#Build the drug viral info network which we want to test
viral_info = pd.read_csv("../data/COVID-19/sars_cov_2.fasta",header='infer',sep=",")
print(viral_info)
drug_info = pd.read_csv("../data/COVID-19/all_keys.list",header=None,sep=",")
drug_info.columns=['standard_inchi_key',"drug_name","canonical_smiles"]

#Only keep drugs whose length is less than 128 
all_drugs = drug_info["canonical_smiles"].values.tolist()
to_keep_drugs_ids, to_keep_drugs = [],[]
for i in range(len(all_drugs)):
    m = Chem.MolFromSmiles(all_drugs[i])
    if m is not None and '.' not in all_drugs[i] and len(all_drugs[i])>=10 and len(all_drugs[i])<=128:
        rev_smiles = Chem.MolToSmiles(m,isomericSmiles=False)
        to_keep_drugs.append(rev_smiles)
        to_keep_drugs_ids.append(i)

        
#The final set of drugs to consider for SARS-COV-2
rev_drug_info = drug_info.iloc[to_keep_drugs_ids,]
print(rev_drug_info)


#Make combinations of viral protein and drugs for testing
viral_df = pd.concat([viral_info]*rev_drug_info.shape[0],ignore_index=True)
drug_df = pd.concat([rev_drug_info]*viral_info.shape[0],ignore_index=True)
interaction_df = pd.concat([viral_df,drug_df],axis=1)
interaction_df['pchembl_value']=0.0

#Write the drug list in form readable for LSTM autoencoder
#autoencoder_drug_info = pd.DataFrame({'src':interaction_df["canonical_smiles"].values.tolist(),
#                                      'trg':interaction_df["canonical_smiles"].values.tolist()})
#autoencoder_drug_info.to_csv("../data/sars_cov_2_drug_info.csv",index=False)

interaction_df.to_csv("../data/COVID-19/sars_cov_2_drug_viral_interactions_to_predict_full_metadata.csv",index=False)
part_interaction_df = interaction_df.iloc[:,[0,3,4,6,7]]
part_interaction_df.to_csv("../data/COVID-19/sars_cov_2_drug_viral_interactions_to_predict.csv",index=False)
# -

rev_drug_info.to_csv("../data/COVID-19/sars_cov_2_drug_info.csv",index=False)

# +
#Load the dataset with drug autoencoder information and match with current interaction df
#part_ls_interaction_df = pd.read_csv("../data/COVID-19/sars_cov_2_drug_viral_interactions_to_predict_with_LS.csv",header='infer',sep=",")

#Load the dataset with protein autoencoder information
#protein_autoencoder_info = pd.read_csv("../data/COVID-19/sars_cov_2_drug_viral_interactions_to_predict_lat.csv",header=None,sep=",")
#protein_ls_dic = {}
#for i in range(64):
#    protein_ls_dic[i] = 'P_LS_'+str(i)
#protein_autoencoder_info.rename(columns=protein_ls_dic,inplace=True)
#protein_autoencoder_info

#Create the dataset with the latent space of drugs and protein sequences
#sars_test_df_with_ls = pd.concat([part_ls_interaction_df,protein_autoencoder_info],axis=1)
#sars_test_df_with_ls.to_csv("../data/COVID-19/sars_cov_2_drug_viral_interactions_to_predict_with_LS_v2.csv",index=False)


# -
#For a given viral protein get ranked list of drugs for a particular ML method
def get_ranked_list(df,proteins,rev_drug_info,protein_mapping_dict,ranked_list_proteins):
    for i in range(len(proteins)):
        temp_df = df[df["uniprot_accession"]==proteins[i]]
        temp_df = temp_df.sort_values(by="predictions",ascending=False)
        temp2_df = pd.merge(temp_df,rev_drug_info,on="standard_inchi_key",how='left')
        drug_info = temp2_df["drug_name"].values.tolist()
        ranked_list_proteins[protein_mapping_dict[proteins[i]]].append(drug_info)
    return(ranked_list_proteins)


# +
#Aggregate the ranked list of drugs to get final set of ordered list of drugs
#def per_protein_rank(ranked_list_proteins, protein_name):
#    temp_list = ranked_list_proteins[protein_name]
#    agg = ra.RankAggregator()
#    return(agg.average_rank(temp_list))

# +
#Use rev_drug_info and results from ML methods to generate ranked list
rf_predictions = pd.read_csv("../results/RF_supervised_sars_cov2_test_predictions.csv",header='infer',sep=",")
svm_predictions = pd.read_csv("../results/SVM_supervised_sars_cov2_test_predictions.csv",header='infer',sep=",")
xgb_predictions = pd.read_csv("../results/XGB_supervised_sars_cov2_test_predictions.csv",header='infer',sep=",")
cnn_predictions = pd.read_csv("../results/cnn_supervised_sars_cov_2_test_predictions.csv",header='infer',sep=",")
lstm_predictions = pd.read_csv("../results/lstm_supervised_sars_cov_2_test_predictions.csv",header='infer',sep=",")
cnn_lstm_predictions = pd.read_csv("../results/cnn_lstm_supervised_sars_cov_2_test_predictions.csv",header='infer',sep=",")
gat_cnn_predictions = pd.read_csv("../results/GAT_model_prediction_on_sars_cov_2.csv",header='infer',sep=',')

#Get a list of the unique proteins
all_proteins = rf_predictions["uniprot_accession"].unique()

#Create a dictionary of ranked list based on protein names
ranked_list_proteins = {}
protein_mapping_dict = {}
for i in range(len(all_proteins)):
    protein_fragment=viral_df[viral_df["uniprot_accession"]==all_proteins[i]]["Protein_Fragment"].unique()
    protein_fragment=protein_fragment[0]
    protein_mapping_dict[all_proteins[i]]=protein_fragment
    ranked_list_proteins[protein_fragment]=[]
    

#Get ranked list for each protein using ML methods except RF
#ranked_list_proteins = get_ranked_list(rf_predictions,all_proteins,rev_drug_info,protein_mapping_dict,ranked_list_proteins)
ranked_list_proteins = get_ranked_list(svm_predictions,all_proteins,rev_drug_info,protein_mapping_dict,ranked_list_proteins)
ranked_list_proteins = get_ranked_list(xgb_predictions,all_proteins,rev_drug_info,protein_mapping_dict,ranked_list_proteins)
ranked_list_proteins = get_ranked_list(cnn_predictions,all_proteins,rev_drug_info,protein_mapping_dict,ranked_list_proteins)
ranked_list_proteins = get_ranked_list(lstm_predictions,all_proteins,rev_drug_info,protein_mapping_dict,ranked_list_proteins)
ranked_list_proteins = get_ranked_list(cnn_lstm_predictions,all_proteins,rev_drug_info,protein_mapping_dict,ranked_list_proteins)
ranked_list_proteins = get_ranked_list(gat_cnn_predictions,all_proteins,rev_drug_info,protein_mapping_dict,ranked_list_proteins)
# -

ranked_list_proteins

# +
#Perform rank aggregation per protein: this ranking strategy is not used
#protein_names=[]
#for i in range(len(all_proteins)):
#    protein_names.append(protein_mapping_dict[all_proteins[i]])
#print(protein_names)

#Get ranked list for each viral protein
#rankings = per_protein_rank(ranked_list_proteins,protein_names[2])
#rankings_df = pd.DataFrame(rankings,columns=['Drug','Overall Weight'])
#rankings_df['Protein_Fragment']=protein_names[2]
#rankings_df.iloc[0:15,:]


# -

#Combine predictions to get rankings based on average predictions
def combined_df(df1,df2,df3,df4,df5,df6,protein_id):
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
    
    temp_df6=df6[df6["uniprot_accession"]==protein_id]
    temp_df6=temp_df6.sort_values(by="standard_inchi_key")
    temp_df6 = temp_df6.reset_index(drop=True)
    
    
    final_df=pd.concat([temp_df1.iloc[:,0:3],temp_df2.iloc[:,2],
                                     temp_df3.iloc[:,2],temp_df4.iloc[:,2],
                                     temp_df5.iloc[:,2], temp_df6.iloc[:,2]],axis=1,join='inner',ignore_index=True)
    return(final_df)


#Combine predictions of models and rank based on average predicted pChEMBL values
def get_results_with_pchembl(final_combined_df,rev_drug_info,protein_name):
    average_combined_df = final_combined_df.iloc[:,[0,1]].copy()
    average_combined_df.columns=["uniprot_accession","standard_inchi_key"]
    average_combined_df["avg_predictions"]=final_combined_df.iloc[:,[2,3,4,5,6]].mean(axis=1)
    final_output_df = pd.merge(average_combined_df,rev_drug_info.iloc[:,[0,1]],on='standard_inchi_key')
    final_output_df["protein_fragment"]=protein_name
    final_output_df = final_output_df.iloc[:,[0,1,4,3,2]]
    final_output_df = final_output_df.sort_values("avg_predictions",ascending=False)
    final_output_df = final_output_df.reset_index(drop=True)
    return(final_output_df)


# +
#For PL-PRO (nsp3)
final_combined_df = combined_df(svm_predictions,xgb_predictions,cnn_predictions,lstm_predictions,cnn_lstm_predictions,gat_cnn_predictions,all_proteins[0])
output1_df = get_results_with_pchembl(final_combined_df,rev_drug_info,protein_mapping_dict[all_proteins[0]])
output1_df.to_csv("../results/PL_Pro_Top_Ranked_Drugs.csv",index=False)

#For 3-CL Pro
final_combined_df = combined_df(svm_predictions,xgb_predictions,cnn_predictions,lstm_predictions,cnn_lstm_predictions,gat_cnn_predictions,all_proteins[1])
output2_df = get_results_with_pchembl(final_combined_df,rev_drug_info,protein_mapping_dict[all_proteins[1]])
output2_df.to_csv("../results/3CL_Pro_Top_Ranked_Drugs.csv",index=False)

#For Spike Protein
final_combined_df = combined_df(svm_predictions,xgb_predictions,cnn_predictions,lstm_predictions,cnn_lstm_predictions,gat_cnn_predictions,all_proteins[2])
output3_df = get_results_with_pchembl(final_combined_df,rev_drug_info,protein_mapping_dict[all_proteins[2]])
output3_df.to_csv("../results/Spike_Pro_Top_Ranked_Drugs.csv",index=False)
# -
output2_df.iloc[0:15,:]
