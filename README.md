# A Consensus of In-silico Sequence-based Modeling Techniques for Compound-Viral Protein Activity Prediction for SARS-COV-2  

Here we provide the scripts, data, model and results for predicting drug-viral activity scores using our pipeline.

Details about how to obtain the training and test set for the predictive models is provided in the data folder.

We consider the SARS-COV-2 virus as a use-case and provide ranked list of drugs based on the 3 main proteases:

a) 3CL-pro  
b) PL-pro  
c) Spike protein  


Code Requirements:  
a) pandas, pickle, numpy, sklearn, scipy all of which are available in latest Anaconda package  
b) xgboost version 0.90, shap   
c) pytorch, torchtext, RDKIT, jupytext, SmilesPE, cuda (for GPU)  


The three traditional machine learning models:

a) Random Forests - scripts/rf_on_ls_protein_drug.py  
b) SVM - scripts/svm_on_ls_protein_drug.py  
c) XGBoost - scripts/svm_on_ls_protein_drug.py

Training done on Train_Drug_Viral_interactions_with_LS_v2_for_Supervised_Learning.csv and available at https://drive.google.com/file/d/1jsYev2WxC0N_OpNCVm-eLhsF_CMeJE9Y/view?usp=sharing  
Testing done on Test_Drug_Viral_interactions_with_LS_v2_for_Supervised_Learning.csv present in gz format in the data folder.

For COVID-19 use case the code is run in test mode on: data/COVID_19/sars_cov_2_drug_viral_interactions_to_predict_with_LS_v2.csv  

Outputs:  
a) RF - results/RF_supervised_test_predictions.csv and results/RF_supervised_sars_cov2_test_predictions.csv  
b) SVM - results/SVM_supervised_test_predictions.csv and results/SVM_supervised_sars_cov2_test_predictions.csv  
c) XGB - results/XGB_supervised_test_predictions.csv and results/XGB_supervised_sars_cov2_test_predictions.csv   


The three end-to-end deep learning models:  

a) CNN - scripts/torchtext_cnn_supervised_learning.py  
b) LSTM - scripts/torchtext_lstm_supervised_learning.py  
c) CNN-LSTM - scripts/torchtext_cnn_lstm_supervised_learning.py  

Runs on test mode:  
1. data/Test_Drug_Viral_interactions_for_Supervised_Learning.csv  
2. data/sars_cov_2_drug_viral_interactions_to_predict.csv

Ouputs:  
a) CNN - results/cnn_supervised_test_predictions.csv and results/cnn_supervised_sars_cov_2_test_predictions.csv  
b) LSTM - results/lstm_supervised_test_predictions.csv and results/lstm_supervised_sars_cov_2_test_predictions.csv  
c) CNN-LSTM - results/cnn_lstm_supervised_test_predictions.csv, results/cnn_lstm_supervised_sars_cov_2_test_predictions.csv

To compare performance of methods on test set:  
a) Install R, Librarires- ggplot2, ggthemes  
b) Run make_error_correlation_plots.R  


To get ranked list of drugs for SARS-COV-2 viral proteins:   
a) Run sars_cov_2_processing.py

Outputs:  
a) 3CL-Pro - results/3CL_Pro_Top_Ranked_Drugs.csv  
b) PL-Pro - results/PL_Pro_Top_Ranked_Drugs.csv  
c) Spike  - results/Spike_Pro_Top_Ranked_Drugs.csv  

