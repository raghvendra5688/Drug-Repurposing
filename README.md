# A Consensus of In-silico Sequence-based Modeling Techniques for Compound-Viral Protein Activity Prediction for SARS-COV-2  

Here we provide the scripts, data, model and results for predicting compound-viral activity scores using our pipeline.

Details about how to obtain the training and test set for the predictive models is provided in the data folder.

We consider the SARS-COV-2 virus as a use-case and provide ranked list of compounds based on the 3 main proteases:

a) 3CL-pro  
b) PL-pro  
c) Spike protein  


An installation guide ("installation_guide.txt") is included in this repository which should complete all system requirements. 


This package contains seven individual machine learning models. The four traditional machine learning models are as follows:

a) GLM - `scripts/supervised_glm_on_ls_protein_compound.py`   
b) Random Forests - `scripts/supervised_rf_on_ls_protein_compound.py`  
c) SVM - `scripts/supervised_svm_on_ls_protein_compound.py`  
d) XGBoost - `scripts/supervised_xgb_on_ls_protein_compound.py`

Training is done on either `Train_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv` with test set `Test_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv` or on `Train_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv` with test set `Test_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv` produced in `data` folder by following the instructions in the **README** available in the `data` folder.

For COVID-19 use case the code is run in test mode on: `data/COVID_19/sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv` and `sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv`

Outputs:  

a) GLM - `results/GLM_MFP_Compound_LS_Proteinsupervised_test_predictions.csv`, `results/GLM_LS_Compound_LS_Proteinsupervised_test_predictions.csv`, `results/GLM_supervised_sars_cov2_test_predictions.csv` 

b) RF - `results/RF_MFP_Compound_LS_Proteinsupervised_test_predictions.csv`, `results/RF_LS_Compound_LS_Proteinsupervised_test_predictions.csv` and `results/RF_supervised_sars_cov2_test_predictions.csv`  

c) SVM - `results/SVM_MFP_Compound_LS_Proteinsupervised_test_predictions.csv`, `results/SVM_LS_Compound_LS_Proteinsupervised_test_predictions.csv` and `results/SVM_supervised_sars_cov2_test_predictions.csv`  

d) XGB - `results/XGB_MFP_Compound_LS_Proteinsupervised_test_predictions.csv`, `results/XGB_LS_Compound_LS_Proteinsupervised_test_predictions.csv` and `results/XGB_supervised_sars_cov2_test_predictions.csv`   


The three end-to-end deep learning models:  

a) CNN - `scripts/torchtext_cnn_supervised_learning.py`  
b) LSTM - `scripts/torchtext_lstm_supervised_learning.py`  
c) CNN-LSTM - `scripts/torchtext_cnn_lstm_supervised_learning.py`  
d) GAT-CNN  - `scripts/torchtext_gat_cnn_supervised_learning.py`

Runs on test mode:  
1. `data/Test_Compound_Viral_interactions_for_Supervised_Learning.csv`  
2. `data/sars_cov_2_Compound_Viral_interactions_for_Supervised_Learning.csv`

Ouputs:  
a) CNN - `results/cnn_supervised_test_predictions.csv` and `results/cnn_supervised_sars_cov_2_test_predictions.csv`  
b) LSTM - `results/lstm_supervised_test_predictions.csv` and `results/lstm_supervised_sars_cov_2_test_predictions.csv`  
c) CNN-LSTM - `results/cnn_lstm_supervised_test_predictions.csv` and `results/cnn_lstm_supervised_sars_cov_2_test_predictions.csv`
d) GAT-CNN - `results/gat_cnn_supervised_test_predictions.csv` and `results/gat_cnn_supervised_sars_cov_2_test_predictions.csv`


To compare the performance of the methods on the test set:  
a) Install R, Librarires- ggplot2, ggthemes  
b) Run `R make_error_correlation_plots.R`  


To get a ranked list of compounds for SARS-COV-2 viral proteins:   
a) Run `sars_cov_2_postprocessing.py`

Outputs:  
a) 3CL-Pro - `results/3CL_Pro_Top_Ranked_Compounds.csv`  
b) PL-Pro - `results/PL_Pro_Top_Ranked_Compounds.csv`  
c) Spike  - `results/Spike_Pro_Top_Ranked_Compounds.csv`  
