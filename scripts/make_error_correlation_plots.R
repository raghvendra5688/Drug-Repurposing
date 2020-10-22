library(data.table)
library(ggplot2)
library(ggthemes)
warnings("off")

setwd(".")

get_error_info <- function(filename)
{
  df <- read.table(filename,header=TRUE,sep=",")
  order_indices <- order(df$labels)
  error_df <- df[order_indices,]
  return(error_df)
}

get_r2 <- function(a,b)
{
  v <- cor(a,b,method="pearson")^2
  return(v)
}
get_mse <- function(a,b)
{
  v <- sum((a-b)^2)/length(a);
  return(v)
}
get_mae <- function(a,b)
{
  v <- sum(abs(a-b))/length(a);
  return(v)
}
get_pearson <- function(a,b)
{
  v <- cor(a,b,method="pearson");
  return(v)
}

get_correlation_plot <- function(df,method)
{
  par(mfrow=c(1,2))
  plot(x=df$labels,y=df$predictions,
       xlab="True Pchembl Values",ylab ="Predicted Pchembl Values",col="red",ylim=c(3,11),xlim=c(3,11))
  abline(0,1,col="black")
  legend(3.2,11, legend=paste0("Linear Fit of ",method),
         col="black",cex=0.8)
  
  df_r2 <- get_r2(df$predictions,df$labels)
  df_mse <- get_mse(df$predictions,df$labels)
  df_mae <- get_mae(df$predictions,df$labels)
  df_pearson <- get_pearson(df$predictions,df$labels)
  text(9,3.0,labels=sprintf("R2=%.3f",df_r2),cex=1.2)
  text(9,3.5,labels=sprintf("Pearson r=%.3f",df_pearson),cex=1.2)
  text(9,4.0,labels=sprintf("MAE=%.3f",df_mae),cex=1.2)
  text(9,4.5,labels=sprintf("MSE=%.3f",df_mse),cex=1.2)
  
  residuals <- df$predictions-df$labels
  h <- hist(residuals,breaks=20,main="",xlab=paste0("Regular Residual of ",method),ylab="Count",freq=T)
  multiplier <- h$counts/h$density
  df_density <- density(residuals)
  df_density$y <- df_density$y*multiplier
  myx <- seq(min(residuals), max(residuals), length.out= 100)
  df_normal <- dnorm(x=myx,mean=mean(residuals),sd=sd(residuals))
  lines(myx, df_normal * multiplier[1], col = "blue", lwd = 2)
}

conver_character <- function(df)
{
  df$uniprot_accession <- as.character(as.vector(df$uniprot_accession))
  df$standard_inchi_key <- as.character(as.vector(df$standard_inchi_key))
  return(df)
}

#Get data frames with errors
glm_smiles_error_df <- get_error_info("../results/GLM_LS_Compound_LS_Protein_supervised_test_predictions.csv")
rf_smiles_error_df <- get_error_info("../results/RF_LS_Compound_LS_Proteinsupervised_test_predictions.csv")
xgb_smiles_error_df <- get_error_info("../results/XGB_LS_Compound_LS_Proteinsupervised_test_predictions.csv")
svm_smiles_error_df <- get_error_info("../results/SVM_LS_Compound_LS_Proteinsupervised_test_predictions.csv")

glm_mfp_error_df <- get_error_info("../results/GLM_MFP_Compound_LS_Protein_supervised_test_predictions.csv")
rf_mfp_error_df <- get_error_info("../results/RF_MFP_Compound_LS_Proteinsupervised_test_predictions.csv")
xgb_mfp_error_df <- get_error_info("../results/XGB_MFP_Compound_LS_Proteinsupervised_test_predictions.csv")
svm_mfp_error_df <- get_error_info("../results/SVM_MFP_Compound_LS_Proteinsupervised_test_predictions.csv")

cnn_error_df <- get_error_info("../results/cnn_supervised_Test_predictions.csv")
lstm_error_df <- get_error_info("../results/lstm_supervised_Test_predictions.csv")
cnn_lstm_error_df <- get_error_info("../results/cnn_lstm_supervised_Test_predictions.csv")
gan_cnn_error_df <- get_error_info("../results/gat_cnn_supervised_Test_predictions.csv")

glm_smiles_error_df <- conver_character(glm_smiles_error_df)
rf_smiles_error_df <- conver_character(rf_smiles_error_df)
svm_smiles_error_df <- conver_character(svm_smiles_error_df)
xgb_smiles_error_df <- conver_character(xgb_smiles_error_df)

glm_mfp_error_df <- conver_character(glm_mfp_error_df)
rf_mfp_error_df <- conver_character(rf_mfp_error_df)
svm_mfp_error_df <- conver_character(svm_mfp_error_df)
xgb_mfp_error_df <- conver_character(xgb_mfp_error_df)

cnn_error_df <- conver_character(cnn_error_df)
lstm_error_df <- conver_character(lstm_error_df)
cnn_lstm_error_df <- conver_character(cnn_lstm_error_df)
gan_cnn_error_df <- conver_character(gan_cnn_error_df)

glm_smiles_error_df <- glm_smiles_error_df[order(glm_smiles_error_df[,1],glm_smiles_error_df[,2]),]
rf_smiles_error_df <- rf_smiles_error_df[order(rf_smiles_error_df[,1],rf_smiles_error_df[,2]),]
svm_smiles_error_df <- svm_smiles_error_df[order(svm_smiles_error_df[,1],svm_smiles_error_df[,2]),]
xgb_smiles_error_df <- xgb_smiles_error_df[order(xgb_smiles_error_df[,1],xgb_smiles_error_df[,2]),]

glm_mfp_error_df <- glm_mfp_error_df[order(glm_mfp_error_df[,1],glm_mfp_error_df[,2]),]
rf_mfp_error_df <- rf_mfp_error_df[order(rf_mfp_error_df[,1],rf_mfp_error_df[,2]),]
svm_mfp_error_df <- svm_mfp_error_df[order(svm_mfp_error_df[,1],svm_mfp_error_df[,2]),]
xgb_mfp_error_df <- xgb_mfp_error_df[order(xgb_mfp_error_df[,1],xgb_mfp_error_df[,2]),]

cnn_error_df <- cnn_error_df[order(cnn_error_df[,1],cnn_error_df[,2]),]
lstm_error_df <- lstm_error_df[order(lstm_error_df[,1],lstm_error_df[,2]),]
cnn_lstm_error_df <- cnn_lstm_error_df[order(cnn_lstm_error_df[,1],cnn_lstm_error_df[,2]),]
gan_cnn_error_df <- gan_cnn_error_df[order(gan_cnn_error_df[,1],gan_cnn_error_df[,2]),]

#Make data frame with predictions
N <- nrow(cnn_error_df)
predictions_df <- data.frame(Method = c(rep("True",N),rep("XGB (SMILES)",N),rep("SVM (MFP)",N),
                                        rep("XGB (MFP)",N),rep("CNN",N),rep("GAT-CNN",N)),
                  Values = c(cnn_error_df$labels,
                             xgb_smiles_error_df$predictions,svm_mfp_error_df$predictions,
                             xgb_mfp_error_df$predictions,cnn_error_df$predictions,
                             gan_cnn_error_df$predictions),
                  Range = c(c(1:N),c(1:N),c(1:N),c(1:N),c(1:N),c(1:N)))

predictions_df$Values <- as.numeric(as.vector(predictions_df$Values))
predictions_df$Range <- as.numeric(as.vector(predictions_df$Range))
sample <- seq(1,N,30)
predictions_df_revised <- predictions_df[predictions_df$Range%in% sample,]

g3 <- ggplot(predictions_df_revised,aes(Range,Values,colour=Method)) + geom_point() +
  geom_smooth(se=FALSE,method=lm,formula=y ~ splines::bs(x, 12))+
  xlab("Test Samples") + ylab("Pchembl Value") + theme_bw() +
  theme(axis.text.x = element_text(size=18),axis.text.y = element_text(size=18),
        axis.title.x = element_text(size=22),
        axis.title.y = element_text(size=22),
        legend.title = element_text(size=18),
        legend.text = element_text(size=16))

#Save the image on disk
ggsave(filename="../results/Fitting_plot_for_pchembl_values.pdf",plot = g3,
       device=pdf(),height=8,width=10,dpi = 300)
dev.off()

pdf(file="../results/GLM_SMILES_Residual_plot_for_pchembl_values.pdf",width=12,height=7,pointsize=16)
get_correlation_plot(glm_smiles_error_df,"GLM (SMILES)")
dev.off()
pdf(file="../results/RF_SMILES_Residual_plot_for_pchembl_values.pdf",width = 12,height=7,pointsize=16)
get_correlation_plot(rf_smiles_error_df,"RF (SMILES)")
dev.off()
pdf(file="../results/SVM_SMILES_Residual_plot_for_pchembl_values.pdf",width = 12,height=7,pointsize=16)
get_correlation_plot(svm_smiles_error_df,"SVM (SMILES)")
dev.off()
pdf(file="../results/XGB_SMILES_Residual_plot_for_pchembl_values.pdf",width = 12,height=7,pointsize=16)
get_correlation_plot(xgb_smiles_error_df,"XGB (SMILES)")
dev.off()

pdf(file="../results/GLM_MFP_Residual_plot_for_pchembl_values.pdf",width=12,height=7,pointsize=16)
get_correlation_plot(glm_mfp_error_df,"GLM (MFP)")
dev.off()
pdf(file="../results/RF_MFP_Residual_plot_for_pchembl_values.pdf",width = 12,height=7,pointsize=16)
get_correlation_plot(rf_mfp_error_df,"RF (MFP)")
dev.off()
pdf(file="../results/SVM_MFP_Residual_plot_for_pchembl_values.pdf",width = 12,height=7,pointsize=16)
get_correlation_plot(svm_mfp_error_df,"SVM (MFP)")
dev.off()
pdf(file="../results/XGB_MFP_Residual_plot_for_pchembl_values.pdf",width = 12,height=7,pointsize=16)
get_correlation_plot(xgb_mfp_error_df,"XGB (MFP)")
dev.off()


pdf(file="../results/CNN_Residual_plot_for_pchembl_values.pdf",width = 12,height=7,pointsize=16)
get_correlation_plot(cnn_error_df,"CNN")
dev.off()
pdf(file="../results/LSTM_Residual_plot_for_pchembl_values.pdf",width = 12,height=7,pointsize=16)
get_correlation_plot(lstm_error_df,"LSTM")
dev.off()
pdf(file="../results/CNN_LSTM_Residual_plot_for_pchembl_values.pdf",width = 12,height=7,pointsize=16)
get_correlation_plot(cnn_lstm_error_df,"CNN-LSTM")
dev.off()
pdf(file="../results/GAT_CNN_Residual_plot_for_pchembl_values.pdf",width = 12,height=7,pointsize=16)
get_correlation_plot(gan_cnn_error_df,"GAT-CNN")
dev.off()
