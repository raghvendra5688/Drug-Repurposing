# A Modelling Framework for Embedding-based Predictions for Compound-Viral Protein Activity

The guidelines to setup the environment and all the packages required to run our framework is available in `installation_guide.md` file

Here we provide the scripts, data, model and results for predicting compound-viral activity scores using our pipeline.

Details about how to obtain the training and test set for the predictive models is provided in the data folder.

We consider the SARS-COV-2 virus as a use-case and provide ranked list of compounds based on the 3 main proteases:

a) 3CL-pro  
b) PL-pro  
c) Spike protein  


An installation guide ("installation_guide.txt") is included in this repository which should complete all system requirements. 

This package contains eight individual machine learning models. The four traditional machine learning models are as follows:

a) GLM - `scripts/supervised_glm_on_ls_protein_compound.py`   
b) Random Forests - `scripts/supervised_rf_on_ls_protein_compound.py`  
c) SVM - `scripts/supervised_svm_on_ls_protein_compound.py`  
d) XGBoost - `scripts/supervised_xgb_on_ls_protein_compound.py`

How to run:

 * `cd scripts`

 * `python supervised_<method>_on_ls_protein_compound.py Train_Compound_Viral_interactions_for_Supervised_Learning_with_<compound_features>_LS.csv <type>_Compound_Viral_interactions_for_Supervised_Learning_with_<compound_features>_LS.csv <method>_<compound_features>_Compound_LS_Protein_supervised_<type>_predictions.csv`

Here `<method>` can be either `glm`, `rf`, `svm`, `xgb`, `<compound_features>` can be either `MFP` or `LS` and `<type>` can either `Test` or `sars_cov_2`.

The files for training and testing (`Test` or `sars_cov_2`) are produced in `data` folder by following the instructions in the **README** available in the `data` folder.

Outputs:  

a) GLM - `results/GLM_MFP_Compound_LS_Protein_supervised_Test_predictions.csv`, `results/GLM_LS_Compound_LS_Protein_supervised_Test_predictions.csv`, `results/GLM_supervised_sars_cov2_predictions.csv` 

b) RF - `results/RF_MFP_Compound_LS_Protein_supervised_Test_predictions.csv`, `results/RF_LS_Compound_LS_Protein_supervised_Test_predictions.csv` and `results/RF_supervised_sars_cov2_predictions.csv`  

c) SVM - `results/SVM_MFP_Compound_LS_Protein_supervised_Test_predictions.csv`, `results/SVM_LS_Compound_LS_Protein_supervised_test_predictions.csv` and `results/SVM_supervised_sars_cov2_predictions.csv`  
d) XGB - `results/XGB_MFP_Compound_LS_Protein_supervised_Test_predictions.csv`, `results/XGB_LS_Compound_LS_Protein_supervised_Test_predictions.csv` and `results/XGB_supervised_sars_cov2_predictions.csv`   


The four end-to-end deep learning models:  

a) CNN - `scripts/torchtext_cnn_supervised_learning.py`  
b) LSTM - `scripts/torchtext_lstm_supervised_learning.py`  
c) CNN-LSTM - `scripts/torchtext_cnn_lstm_supervised_learning.py`  
d) GAT-CNN  - `scripts/torchtext_gat_cnn_supervised_learning.py`

Runs on test mode:  
1. `data/Test_Compound_Viral_interactions_for_Supervised_Learning.csv`  
2. `data/sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning.csv`

How to run:

 * `cd scripts`

 * `python torchtext_<method>_supervised_learning.py Train_Compound_Viral_interactions_for_Supervised_Learning.csv <type>_Compound_Viral_interactions_for_Supervised_Learning.csv <method>_supervised_<type>_predictions.csv`

Here `<method>` can be either `cnn`, `lstm`, `cnn_lstm`, `gat_cnn` and `<type>` can be either `Test` or  `sars_cov_2`.

Ouput Files with location:

a) CNN - `results/cnn_supervised_Test_predictions.csv` and `results/cnn_supervised_sars_cov_2_predictions.csv`  
b) LSTM - `results/lstm_supervised_Test_predictions.csv` and `results/lstm_supervised_sars_cov_2_predictions.csv`  
c) CNN-LSTM - `results/cnn_lstm_supervised_Test_predictions.csv` and `results/cnn_lstm_supervised_sars_cov_2_predictions.csv`
d) GAT-CNN - `results/gat_cnn_supervised_Test_predictions.csv` and `results/gat_cnn_supervised_sars_cov_2_predictions.csv`


To compare the performance of the methods on the test set:  
a) Install R, Librarires- ggplot2, ggthemes  
b) Run `R make_error_correlation_plots.R`  


To get a ranked list of compounds for SARS-COV-2 viral proteins:   
a) Run `sars_cov_2_postprocessing.py`

Outputs:  
a) 3CL-Pro - `results/3CL_Pro_Top_Ranked_Compounds.csv`  
b) PL-Pro - `results/PL_Pro_Top_Ranked_Compounds.csv`  
c) Spike  - `results/Spike_Pro_Top_Ranked_Compounds.csv`  
