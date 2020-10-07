# A Consensus of In-silico Sequence-based Modeling Techniques for Compound-Viral Protein Activity Prediction for SARS-COV-2  

Here we provide the details of the steps followed to prepare the data for training set, test set and sars-cov-2 dataset.

# SMILES Autoencoder

1. We first collected 556,134 SMILES strings for compounds used in the paper [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111) and combined it with 1,936,962 SMILES strings obtained from MOSES dataset [Molecular Sets (MOSES): A BenchmarkingPlatform for Molecular Generation Models](https://github.com/molecularsets/moses). This file is present in the SMILES_Autoencoder/all_smiles.csv

2. We next run: `cd SMILES_Autoencoder; python cleanup_smiles.py all_smiles.csv all_smiles_revised.csv` to filter out compounds containing salts and remove stereochemical information. We also filter compounds based on their length, restricting the final set to include compounds whose SMILES sequence lengths are in [10,128] allowing small sized compounds as well as large size ligands to be part of our chemical search space which is relatively bigger than that used in [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111).

3. The resulting all_smiles_revised.csv contains 2,454,665 compounds.

4. We next run: `python prepare_smiles_autoencoder.py` to obtain all_smiles_revised_final.csv

5. To train the SMILES autoencoder model we run: `cd ../../scripts/; python torchtext_lstm_run.py`. This results in `torchtext_checkpoint.pt` in the models folder.



# Compound Virus Activities for End-to-End Deep Learning Models

We perform search on Pubmed (NCBI) to generate a good AID (Assay Id) list:

1. Protein target GI73745819 - SARS Protease - Called SARS_C3_Assays.txt in this report

2. Protein target GI75593047 - HIV pol polyprotein - Called HIV_Protease_Assays.txt in this report

3. NS3 - Hep3 protease - Called NS3_Protease_Assays.txt in this report

4. 3CL-Pro - Mers Protease - Called MERS_Protease_Assays.txt in this report

All these data are available inside the `additional_data` folder in the `Compound_Virus_Interactions` folder

We obtain the corresponding viral proteases for these viruses through Uniprot and maintain them in `Compound_Virus_Interactions/ncbi_Filtered_Viral_Proteins.csv` file

We run: `cd Compound_Virus_Interactions; gunzip additional_data/ns3_assays.pkl.gz ; python Preprocessing_More_Data.py` to obtain `ncbi_Filtered_Compound_Viral_Proteins_Network.csv`, `ncbi_Filtered_Compounds.csv` and `ncbi_compound_src_target_info.csv` files inside the `Compound_Virus_Interactions` folder.

5. We also download curated compound-viral protein activites available in ChEMBL as `Compound_Viral_protein_Networks.csv`

6. Viral protein sequences are downloaded from Uniprot and are available at (https://drive.google.com/file/d/1nmqUZd5_RKxF_FJ9A_nkHIA9H0sevBMK/view?usp=sharing) which can be downloaded and should be put in the folder `Compound_Virus_Interactions`.

We run: `python chembl_filter_compound_virus_interactions.py` to filter compound-virus activities to be either IC50, Ki or Kd and remove interactions where compound SMILES strings are length either >128 or <10 and compounds contain salts. 

We obtain `chembl_Filtered_Compound_Viral_Proteins_Network.csv`, `chembl_Filtered_Compounds.csv` and `chembl_compound_src_target_info.csv` as a result inside the `Compound_Virus_Interactions` folder.

7. We next run: `cd ../../scripts/ ;  python divide_train_valid_test_deep_learning.py` resulting in training file `Train_Compound_Viral_interactions_for_Supervised_Learning.csv` and test file `Test_Compound_Viral_interactions_for_Supervised_Learning.csv` in the `data` folder.


# Compound Virus Activties for Traditional Supervised Learning Models based on Embeddings
