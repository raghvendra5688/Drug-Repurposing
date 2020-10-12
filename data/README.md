# A Consensus of In-silico Sequence-based Modeling Techniques for Compound-Viral Protein Activity Prediction for SARS-COV-2  

Here we provide the details of the steps followed to prepare the data for training set, test set and sars-cov-2 dataset.

# SMILES Autoencoder

1. We first collected 556,134 SMILES strings for compounds used in the paper [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111) and combined it with 1,936,962 SMILES strings obtained from MOSES dataset [Molecular Sets (MOSES): A BenchmarkingPlatform for Molecular Generation Models](https://github.com/molecularsets/moses). This file is present in the `SMILES_Autoencoder/all_smiles.csv`

2. We next run: `cd SMILES_Autoencoder; python cleanup_smiles.py all_smiles.csv all_smiles_revised.csv` to filter out compounds containing salts and remove stereochemical information. We also filter compounds based on their length, restricting the final set to include compounds whose SMILES sequence lengths are in [10,128] allowing small sized compounds as well as large size ligands to be part of our chemical search space which is relatively bigger than that used in [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111).

3. The resulting `all_smiles_revised.csv` contains 2,454,665 compounds.

4. We next run: `python prepare_smiles_autoencoder.py` to obtain `all_smiles_revised_final.csv`

5. To train the SMILES autoencoder model we run: `cd ../../scripts/; python torchtext_smiles_autoencoder.py`. This results in `torchtext_checkpoint.pt` in the models folder.



# Compound Virus Activities for End-to-End Deep Learning Models

We perform search on Pubmed (NCBI) to generate a good AID (Assay Id) list:

1. Protein target GI73745819 - SARS Protease - Called `SARS_C3_Assays.txt` in this report

2. Protein target GI75593047 - HIV pol polyprotein - Called `HIV_Protease_Assays.txt` in this report

3. NS3 - Hep3 protease - Called `NS3_Protease_Assays.txt` in this report

4. 3CL-Pro - Mers Protease - Called `MERS_Protease_Assays.txt` in this report

All these data are available inside the `additional_data` folder in the `Compound_Virus_Interactions` folder

We obtain the corresponding viral proteases for these viruses through Uniprot and maintain them in `Compound_Virus_Interactions/ncbi_Filtered_Viral_Proteins.csv` file

We run: `cd Compound_Virus_Interactions; gunzip additional_data/ns3_assays.pkl.gz ; python Preprocessing_More_Data.py` to obtain `ncbi_Filtered_Compound_Viral_Proteins_Network.csv` files inside the `Compound_Virus_Interactions` folder.

5. We also download curated compound-viral protein activites available in ChEMBL as `Compound_Viral_protein_Networks.csv`

6. Viral protein sequences are downloaded from Uniprot and are available at (https://drive.google.com/file/d/1nmqUZd5_RKxF_FJ9A_nkHIA9H0sevBMK/view?usp=sharing) which can be downloaded and should be put in the folder `Compound_Virus_Interactions`.

We run: `python chembl_filter_compound_virus_interactions.py` to filter compound-virus activities to be either IC50, Ki or Kd and remove interactions where compound SMILES strings are length either >128 or <10 and compounds contain salts. 

We obtain `chembl_Filtered_Compound_Viral_Proteins_Network.csv` as a result inside the `Compound_Virus_Interactions` folder.

7. We next run: `cd ../../scripts/ ;  python train_valid_test_deep_learning.py` which combines the `chembl_Filtered_Compound_Viral_Proteins_Network.csv` and `ncbi_Filtered_Compound_Viral_Proteins_Network.csv` resulting in training file `Train_Compound_Viral_interactions_for_Supervised_Learning.csv` and test file `Test_Compound_Viral_interactions_for_Supervised_Learning.csv` in the `data` folder.


# Compound Virus Activties for Traditional Supervised Learning Models based on Embeddings

To generate the embedding representation of the compounds using the Teacher Forcing-LSTM SMILES Autoencoder, we need to follow the below mentioned steps:

1. For training set, run `cd scripts` from home folder of repository and then run `python ls_generator_smiles.py --input Train_Compound_Viral_interactions_for_Supervised_Learning.csv --output Train_Compound_LS.csv`. This will result in the file `Train_Compound_LS.csv` in the `data` folder with same number of lines as in training file and 256 dimensions.

2. For test set run `python ls_generator_smiles.py --input Test_Compound_Viral_interactions_for_Supervised_Learning.csv --output Test_Compound_LS.csv` to obtain `Test_Compound_LS.csv` in the `data` folder.


To generate the vector representation of the compounds using Morgan Fingerprints, we need to follow the below mentioned steps:

1. For training set, run `cd scripts` from home of repository and then run `python ls_generator_morgan.py --input Train_Compound_Viral_interactions_for_Supervised_Learning.csv --output Train_Compound_MFP.csv`. This will result in the file `Train_Compound_MFP.csv` in the `data` folder with the same number of lines as training file and 256 dimensions.

2. For test set run `python ls_generator_morgan.py --input Test_Compound_Viral_interactions_for_Supervised_Learning.csv --output Test_Compound_MFP.csv` to obtain `Test_Compound_MFP.csv` in the `data` folder.


To generate the vector representation of the viral proteins, we need to follow the below mentioned steps:




To produce the training and test set with vector representation of compounds from SMILES autoencoder/Morgan Fingerprints and latent space representation of viral proteins, we take the following steps:

1. Run `cd scripts`

2. Run `python train_valid_test_supervised learning_on_ls.py` to produce `Train_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv` and `Test_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv` for SMILES autoencoder + protein autoencoder embedding combination for train and test set respectively. Similarly, we also obtain `Train_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv` and `Test_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv` for Morgan fingerprints + protein autoencoder embedding combination for train and test set respectively. All these files are produced in the `data` folder.


# Compound Virus Activities Prediction for SARS-COV-2 Viral proteins

1. We downloaded the list of compounds from [Drug Virus Info](http://drugvirus.info/), [Barabasi paper](https://arxiv.org/abs/2004.07229), [Discovery of SARS-CoV-2 antiviral drugs through large-scale compound repurposing](https://www.nature.com/articles/s41586-020-2577-1) which we put in the `data/COVID-19/compound_virus_info.lst`, `data/COVID-19/compounds_barabasi.list` and `data/COVID-19/compounds_12k.csv` files respectively. The 12k compounds mentioned in (https://www.nature.com/articles/s41586-020-2577-1) are available in the [ReframeDB](https://reframedb.org/). On the reframedb website the authors don't provide the name of the compounds rather provide information about 68 different assays which are available for download. Upon requesting the name of compounds, the authors suggested to download the assays and use all the compounds available in the assays. As a result we end up with 2383 compounds out of the 12k compounds mentioned in the paper. 

2. We combine all these compounds by running `cd data/COVID-19; python get_all_compounds_COVID_19.py` and screen for unique compounds whose SMILES strings are 10<=length(x)<=128 and don't contain salt bridges resulting in a total of 1663 compounds in `all_verified_keys.list`.

3. We prepare the SARS-COV-2 test set for end-to-end deep learning by running `python sars_cov_2_preprocessing.py` from the `data/COVID-19` folder resulting in `sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning.csv` file in the `data` folder.

4. We can obtain results for CNN, LSTM and CNN+LSTM models for SARS-COV-2 viral proteins by running `cd scripts` from home directory followed by `torchtext_cnn_supervised_learning.py`, `torchtext_lstm_supervised_learning.py` and `torchtext_cnn_lstm_supervised_learning.py` respectively. More details are available in the README in the home directory.



