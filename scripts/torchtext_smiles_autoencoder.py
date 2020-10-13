#!/usr/bin/python -u
# +
from SeqDataset import *
import tokeniser
import numpy as np
import argparse
import random
import math
import time
#import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import softmax, relu, selu, elu
from torchtext.data import Field, BucketIterator, TabularDataset
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
#Get the src and target fields
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

#Get the data
datafields = [("src", src), # we won't be needing the id, so we pass in None as the field
              ("trg", trg)]
data = TabularDataset(
           path="../data/SMILES_Autoencoder/all_smiles_revised_final.csv", # the file path
           format='csv',
           skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
           fields=datafields)

train_data, valid_data = data.split(split_ratio=0.8)
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
del data
torch.cuda.empty_cache()   

# +
print(vars(valid_data.examples[0]))
src.build_vocab(train_data, min_freq = 2)
trg.build_vocab(train_data, min_freq = 2)

print(f"Unique tokens in source vocabulary: {len(src.vocab)}")
print(f"Unique tokens in target vocabulary: {len(trg.vocab)}")
TRG_PAD_IDX = trg.vocab.stoi[trg.pad_token]
print("Padding Id: ",TRG_PAD_IDX)
# -

print(src.vocab.stoi)

# +
BATCH_SIZE = 2048

train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data),
    batch_size = BATCH_SIZE,
    device = DEVICE,
    sort = False,
    shuffle = True
    )

for i,batch in enumerate(valid_iterator):
    print("Train Src Shape: ",str(batch.src.shape))
    print("Train Trg Shape: ",str(batch.trg.shape))

# +
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
# -

optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = trg.vocab.stoi[trg.pad_token]
criterion = nn.CrossEntropyLoss().to(DEVICE)
print(TRG_PAD_IDX)     

# +
N_EPOCHS = 1000
CLIP = 1
counter = 0
patience = 200
train_loss_list = []
valid_loss_list = []
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    if (counter<patience):
        print("Counter Id: ",str(counter))
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        if valid_loss < best_valid_loss:
            counter = 0
            print("Current Val. Loss: %.3f better than prev Val. Loss: %.3f " %(valid_loss,best_valid_loss))
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), '../models/torchtext_checkpoint.pt')
        else:
            counter+=1
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


#model.load_state_dict(torch.load('../models/torchtext_checkpoint.pt'))
valid_loss = evaluate(model, valid_iterator, criterion)
print(f'| Best Valid Loss: {valid_loss:.3f} | Best Valid PPL: {math.exp(valid_loss):7.3f} |')

fout=open("../results/smiles_variables_loss_plot.csv","w")
for i in range(len(train_loss_list)):
    outputstring = str(train_loss_list[i])+","+str(valid_loss_list[i])+"\n"
    fout.write(outputstring)
fout.close()

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


