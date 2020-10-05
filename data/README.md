# Drug Repurposing for COVID-19

Here we provide the details of the steps followed to prepare the data for training set, test set and sars-cov-2 dataset.

# SMILES Autoencoder

1. We first collected 556,134 SMILES strings for compounds used in the paper [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111) and combined it with 1,936,962 SMILES strings obtained from MOSES dataset [Molecular Sets (MOSES): A BenchmarkingPlatform for Molecular Generation Models](https://github.com/molecularsets/moses). This file is present in the SMILES_Autoencoder/all_smiles.csv

2. We next run: `python cleanup_smiles.py all_smiles.csv all_smiles_revised.csv` to filter out compounds containing salts and remove stereochemical information. We also filter compounds based on their length, restricting the final set to include compounds whose SMILES sequence lengths are in [10,128] allowing small sized compounds as well as large size ligands to be part of our chemical search space which is relatively bigger than that used in [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111).

3. The resulting all_smiles_revised.csv contains 2,459,645 compounds.

4. We next run: `python prepare_smiles_autoencoder.py` to obtain all_smiles_revised_final.csv

5. To train the SMILES autoencoder model we run: `cd ../../scripts/; python torchtext_lstm_run.py`. This results in `torchtext_checkpoint.pt` in the models folder.


