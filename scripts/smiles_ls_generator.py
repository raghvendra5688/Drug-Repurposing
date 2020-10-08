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

# +
#Load all pre-requisites
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import softmax, relu, selu, elu
from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import random
import math
import time
from torchtext.datasets import TranslationDataset, Multi30k
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tokeniser import tokenize_drug
from seq2seq import Encoder, Decoder, Seq2Seq, init_weights, count_parameters, train, evaluate, epoch_time

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.is_available()
cudaid = int(0)
DEVICE = torch.device("cuda:%d" % (cudaid) if torch.cuda.is_available() else "cpu")
print(DEVICE)

# +
#Define the src and target for torchtext to process
src = Field(sequential=True,
            tokenize = tokenize_drug, 
            init_token = '<sos>', 
            eos_token = '<eos>'
           )

trg = Field(sequential=True,
            tokenize = tokenize_drug, 
            init_token = '<sos>', 
            eos_token = '<eos>'
           )
# -

#Get the train and test set in torchtext format
datafields = [("src", src), # we won't be needing the id, so we pass in None as the field
              ("trg", trg)]
train = TabularDataset.splits(
        path='../data/SMILES_Autoencoder/', train='all_smiles_revised_final.csv',
        #test='SMILES_to_Representation_v2.csv', 
        format='csv',
        skip_header=True,
        fields=datafields)

#Split the dataset into train and validation set
train_data,valid_data = train[0].split(split_ratio=0.8)

# +
print(f"Number of examples: {len(train_data.examples)}")
src.build_vocab(train_data, min_freq = 2)
trg.build_vocab(train_data, min_freq = 2)
 
#Total no of unique words in our vocabulary
print(f"Unique tokens in source vocabulary: {len(src.vocab)}")
print(f"Unique tokens in target vocabulary: {len(trg.vocab)}")
TRG_PAD_IDX = trg.vocab.stoi[trg.pad_token]
print("Padding Id: ",TRG_PAD_IDX)
 
#Create the iterator to traverse over test samples for which we need to generate latent space
BATCH_SIZE = 128
(train_iterator, test_iterator) = BucketIterator.splits((train_data,valid_data),
                                             batch_size = BATCH_SIZE,
                                             device = DEVICE,
                                             sort = False,
                                             shuffle = False)
print(src.vocab.stoi)
print(trg.vocab.stoi)

# +
#Define the model once again
INPUT_DIM = len(src.vocab)
OUTPUT_DIM = len(trg.vocab)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 1
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device=DEVICE).to(DEVICE)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss().to(DEVICE)
model.load_state_dict(torch.load('../models/lstm_out/torchtext_checkpoint.pt',map_location=torch.device('cpu')))

# +
#Get latent space for all drugs
model.eval()
epoch_loss = 0
    
ls_list = []
encode_list = []
decode_list = []
error_list = []
with torch.no_grad():
    for i, batch in enumerate(test_iterator):
        new_src = batch.src
        new_trg = batch.trg

        #Get output
        outputs = model(new_src, new_trg, 1) #turn on teacher forcing
        output = outputs[0]
        hidden = outputs[1]
        cell_state = outputs[2]

        #Get latent space
        o1 = torch.argmax(torch.softmax(output,dim=2),dim=2)
        h1 = torch.mean(hidden,dim=0).cpu().detach().tolist()
        c1 = torch.mean(cell_state,dim=0).cpu().detach().tolist()
        
        for i in range(len(h1)):
            temp_ls = h1[i]
            temp_encode = new_trg[:,i].cpu().detach().tolist()
            temp_decode = o1[:,i].cpu().detach().tolist()
            try:
                index_1 = temp_decode.index(1)
            except:
                index_1 = len(temp_decode)
            temp_error = np.array(temp_encode)-np.array(temp_decode)
            error = sum(np.absolute(temp_error[1:index_1])>0)/len(temp_error)
            error_list.append(error)
            ls_list.append(temp_ls)
            encode_list.append(temp_encode)
            decode_list.append(temp_decode)
            
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        rev_trg = new_trg[1:].view(-1)
        
        loss = criterion(output, rev_trg)
        print(round(loss.item(),3))
        epoch_loss += loss.item()        
        
#Print overall average error
print(epoch_loss/len(test_iterator));
torch.cuda.empty_cache()
# -

final_list, only_smiles_list =[],[]
for i in range(len(encode_list)):
    temp_encode = encode_list[i]
    temp_decode = decode_list[i]
    temp_encode_str,temp_decode_str, temp_mol_str, temp_error_str = '','','',''
    
    #Get original string
    for j in range(1,len(temp_encode)):
        
        #Break when it sees padding
        if (temp_encode[j]==1):
            break
        
        #Don't pad end of sentence
        if (temp_encode[j]!=3):
            temp_encode_str+=src.vocab.itos[temp_encode[j]]
    
    #Get decoded string
    for j in range(1,len(temp_decode)):
        
        if (temp_decode[j]==1):
            break;
            
        if (temp_decode[j]!=3):
            temp_decode_str+=src.vocab.itos[temp_decode[j]]
            
    
    m = Chem.MolFromSmiles(temp_decode_str)
    if (m is not None):
        temp_mol_str = '1'
    else:
        temp_mol_str = '0'
    
    string_list = [temp_encode_str, temp_decode_str, temp_mol_str, str(error_list[i])]
    only_smiles_list.append(string_list)
    string_list_with_ls = string_list + ls_list[i]
    final_list.append(string_list_with_ls)


final_list[0]

# +
#Create the final dataset with protein sequences and drug latent space
colids=['standard_inchi_key','canonical_smiles','recon_canonical_smiles','is_mol','rec_error'] + ['LS_'+str(x) for x in range(len(ls_list[0][0:256]))]
final_out_df = pd.DataFrame(final_list, columns = colids)
final_out_df = final_out_df.drop_duplicates()
ncols = final_out_df.shape[1]

##drug_viral_protein_info = pd.merge(subset3_drug_viral_protein_info,
#drug_viral_protein_info = pd.merge(test_interaction_info,
#                                           final_out_df,
#                                           on=['standard_inchi_key','canonical_smiles'], how='left')
#drug_viral_protein_info.to_csv("../data/Drug_Protein_Networks/Filtered_Drug_Viral_proteins_Network_with_Drug_LS.csv",index=False)
#drug_viral_protein_info.to_csv("../data/COVID-19/sars_cov_2_additional_drug_viral_interactions_to_predict_with_LS.csv",index=False)
final_out_df.to_csv("./data/SMILES_to_Representation_with_LS.csv",index=False)