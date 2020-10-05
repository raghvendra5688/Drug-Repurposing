#!/usr/bin/python -u
# +
from SeqDataset import *
#import pandas as pd
import numpy as np
import argparse
import random
import math
import time
#import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import softmax, relu, selu, elu
from torchtext.data import Field, LabelField, BucketIterator, TabularDataset, Pipeline
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import random
import math
from torchtext.datasets import TranslationDataset, Multi30k
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tokeniser import tokenize_drug, tokenize_protein
from seq2func import LSTM_Encoder, CNN_Encoder, Seq2Func, init_weights, count_parameters, train, evaluate, epoch_time

SEED = 123
random.seed(SEED)
st = random.getstate()
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.is_available()
cudaid = int(0)
DEVICE = torch.device("cuda:%d" % (cudaid) if torch.cuda.is_available() else "cpu")
print(DEVICE)
# +
#Histogram of pchembl values
#plt.hist(interaction_df["pchembl_value"],bins=10)
# -

#Define all the variables to be read by torchtext TabularDataset
TEXT1         =  Field(sequential=True,
                          tokenize = tokenize_protein,
                          init_token=None,
                          eos_token=None,
                          pad_first=False
                         )
TEXT2         =  Field(sequential=True,
                          tokenize = tokenize_drug,
                          init_token=None,
                          eos_token=None
                         )
LABEL         =  Field(sequential=False, 
                          use_vocab=False, 
                          is_target=True,
                          dtype = torch.float,
                          preprocessing=Pipeline(lambda x: float(x)))
INDEX1        =  Field(sequential=False,use_vocab=True)
INDEX2        =  Field(sequential=False,use_vocab=True)

#Read the data and get Protein Sequence, Canonical Smiles and Pchembl_value
datafields = [('uniprot_accession',INDEX1),
              ("Sequence", TEXT1), 
              ('standard_inchi_key',INDEX2),
              ("canonical_smiles", TEXT2),
              ("pchembl_value",LABEL)
             ]

#Model in test mode for the SARS-COV-2 viral proteins
full_data, data, test_data = TabularDataset.splits(
           path="../data/", train='all_drug_viral_interactions_for_supervised_learning.csv',
           validation='Train_Drug_Viral_interactions_for_Supervised_Learning.csv',
           #test='Test_Drug_Viral_interactions_for_Supervised_Learning.csv',
           test='sars_cov_2_drug_viral_interactions_to_predict.csv',
           format='csv',
           skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
           fields=datafields)

#See how the data looks like
print("Uniprot: ",data.examples[0].uniprot_accession)
print("Protein Sequence: ",data.examples[0].Sequence)
print("Canonical Smiles: ",data.examples[0].canonical_smiles)
print("Inchi Key:",data.examples[0].standard_inchi_key)
print("Target value: ",data.examples[0].pchembl_value)

#Split the data randomly into train, valid and test
train_data, valid_data = data.split(split_ratio=0.9,random_state=random.setstate(st))
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of test examples: {len(test_data.examples)}")
#del data
torch.cuda.empty_cache()   

# +
#Build the sequence and smiles 
TEXT1.build_vocab(train_data, min_freq = 2)
TEXT2.build_vocab(train_data, min_freq = 1)
LABEL.build_vocab(train_data, min_freq = 1)
INDEX1.build_vocab(full_data, min_freq = 1)
INDEX2.build_vocab(full_data, min_freq = 1)

print(f"Unique tokens in Sequence vocabulary: {len(TEXT1.vocab)}")
print(f"Unique tokens in SMILES vocabulary: {len(TEXT2.vocab)}")
print(f"Unique tokens in LABELs vocabulary: {len(LABEL.vocab)}")

SEQUENCE_PAD_IDX = TEXT1.vocab.stoi[TEXT1.pad_token]
print("Padding Id in Sequence: ",SEQUENCE_PAD_IDX)
SMILES_PAD_IDX = TEXT2.vocab.stoi[TEXT2.pad_token]
print("Padding Id in SMILES: ",SMILES_PAD_IDX)

print("Tokens in Sequence vocabulary: ",TEXT1.vocab.stoi)
print("Tokens in SMILES vocabulary: ",TEXT2.vocab.stoi)

# +
BATCH_SIZE = 256

train_iterator, valid_iterator , test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = DEVICE,
    sort = True,
    shuffle = False,
    sort_key = lambda x: len(x.Sequence),
    sort_within_batch = True
    )

for i,batch in enumerate(train_iterator):
    print("Valid Sequence Shape: ",str(batch.Sequence.shape))
    print("Valid canonical_smiles Shape: ",str(batch.canonical_smiles.shape))
    print("Valid pchembl Shape: ",str(batch.pchembl_value.shape))

# +
PROTEIN_INPUT_DIM = len(TEXT1.vocab)
PROTEIN_ENC_EMB_DIM = 64
PROTEIN_OUT_DIM = 256
PAD_IDX1 = TEXT1.vocab.stoi[TEXT1.pad_token]

SMILES_INPUT_DIM = len(TEXT2.vocab)
SMILES_ENC_EMB_DIM = 64
SMILES_OUT_DIM = 256
PAD_IDX2 = TEXT2.vocab.stoi[TEXT2.pad_token]

N_FILTERS = 128
FILTER_SIZES = [3,6,9,12]

HID_DIM = 128
OUT_DIM = 1
DROPOUT = 0.0


protein_enc = CNN_Encoder(PROTEIN_INPUT_DIM, PROTEIN_ENC_EMB_DIM, PROTEIN_OUT_DIM, N_FILTERS, FILTER_SIZES, DROPOUT, PAD_IDX1)
smiles_enc = CNN_Encoder(SMILES_INPUT_DIM, SMILES_ENC_EMB_DIM, SMILES_OUT_DIM, N_FILTERS, FILTER_SIZES, DROPOUT, PAD_IDX2)

model = Seq2Func(protein_enc, smiles_enc, HID_DIM, OUT_DIM, DROPOUT, device=DEVICE).to(DEVICE)
print("Total parameters in model are: ",count_parameters(model))
model.apply(init_weights)
# -

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss().to(DEVICE)

# +
#N_EPOCHS = 1000
#CLIP = 1
#counter = 0
#patience = 400
#train_loss_list = []
#valid_loss_list = []
#best_valid_loss = float('inf')
#for epoch in range(N_EPOCHS):
#    if (counter<patience):
#        print("Counter Id: ",str(counter))
#        start_time = time.time()
#        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#        valid_loss = evaluate(model, valid_iterator, criterion)
#        end_time = time.time()
#        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
#        train_loss_list.append(train_loss)
#        valid_loss_list.append(valid_loss)
#        if valid_loss < best_valid_loss:
#            counter = 0
#            print("Current Val. Loss: %.3f better than prev Val. Loss: %.3f " %(valid_loss,best_valid_loss))
#            best_valid_loss = valid_loss
#            torch.save(model.state_dict(), 'cnn_out/cnn_supervised_checkpoint.pt')
#        else:
#            counter+=1
#        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model.load_state_dict(torch.load('../models/cnn_out/cnn_supervised_checkpoint.pt'))
valid_loss = evaluate(model, valid_iterator, criterion)
print(f'| Best Valid Loss: {valid_loss:.3f} | Best Valid PPL: {math.exp(valid_loss):7.3f} |')

#test_loss = evaluate(model, test_iterator, criterion)
#print(f'| Test Loss: {test_loss: .3f} | Best Test PPL: {math.exp(test_loss):7.3f} |')

#fout=open("cnn_out/cnn_supervised_loss_plot.csv","w")
#for i in range(len(train_loss_list)):
#    outputstring = str(train_loss_list[i])+","+str(valid_loss_list[i])+"\n"
#    fout.write(outputstring)
#fout.close()

# +
model.eval()
output_list = []
uniprot_info = np.array(INDEX1.vocab.itos)
inchikey_info = np.array(INDEX2.vocab.itos)
with torch.no_grad():
    for i, batch in enumerate(test_iterator):
    
        protein_src = batch.Sequence
        smiles_src = batch.canonical_smiles
        trg = batch.pchembl_value
    
        batch_size = trg.shape[0]
        output = model(protein_src, smiles_src).squeeze(1) 
        #output = [batch size]
            
        predictions = output.cpu().detach().tolist()
        uniprot_batch = uniprot_info[batch.uniprot_accession.cpu().detach().tolist()]
        inchikey_batch = inchikey_info[batch.standard_inchi_key.cpu().detach().tolist()]
        label = trg.cpu().detach().tolist()

        for i in range(batch_size):
            temp_list = [uniprot_batch[i],inchikey_batch[i],predictions[i],label[i]]
            output_list.append(temp_list)

        del protein_src
        del smiles_src
        del trg
        torch.cuda.empty_cache()

fout = open("../results/cnn_supervised_sars_cov_2_test_predictions.csv","w")
header = 'uniprot_accession,'+'standard_inchi_key,'+'predictions,'+'labels'+'\n'
fout.write(header)
for data in output_list:
    string_list = [str(x) for x in data];
    temp_str = ",".join(string_list)
    fout.write(temp_str+"\n")

fout.close()
#test_df = pd.DataFrame(output_list, columns=['uniprot_accession','standard_inchi_key','predictions','labels'])
#test_df.to_csv("./cnn_out/cnn_supervised_test_predictions.csv",index=False)
#test_df

# +
# visualize the loss as the network trained
# fig = plt.figure(figsize=(10,8))
# plt.plot(range(1,len(train_loss_list)+1),train_loss_list, label='Training Loss')
# plt.plot(range(1,len(valid_loss_list)+1),valid_loss_list,label='Validation Loss')

# find position of lowest validation loss
# minposs = valid_loss_list.index(min(valid_loss_list))+1 
# plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.ylim(0, 5) # consistent scale
# plt.xlim(0, len(train_loss_list)+1) # consistent scale
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# fig.savefig('lstm_out/smiles_loss_plot.png', bbox_inches='tight')
# -


