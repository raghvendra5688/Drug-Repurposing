from torch.nn.functional import softmax, relu, selu, leaky_relu, elu, max_pool1d
import torch.nn.init as init
import time
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import random
import math
#import spacy
cudaid = int(0)
DEVICE = torch.device("cuda:%d" % (cudaid) if torch.cuda.is_available() else "cpu")
print(DEVICE)


class LSTM_Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, out_dim, n_layers,  dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        
        self.out_dim = out_dim
        
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        #self.relu = leaky_relu
        self.relu = selu
        
        self.fc = nn.Linear(n_layers*hid_dim, out_dim)

    def forward(self, src, batch_size):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        hidden = hidden.permute(1,0,2)
        
        hidden = torch.reshape(hidden,[batch_size,self.n_layers*self.hid_dim])
        
        output = self.dropout(self.fc(hidden))
        
        return output


class CNN_Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, out_dim, n_filters, filter_sizes, dropout, pad_idx):
        super().__init__()

        
        self.out_dim = out_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = emb_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        
        self.dropout = nn.Dropout(dropout)
       
        self.batchnorm = nn.BatchNorm1d(out_dim)
        
        #self.relu = leaky_relu
        self.relu = selu
        
        self.maxpool = max_pool1d
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, out_dim)

    def forward(self, src, batch_size):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]
        
        embedded = embedded.permute(1, 2, 0)
        #embedded = [batch size, emb dim, src len]
        
        conved = [self.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch size, n_filters, src len - filter_sizes[n] + 1]
        
        pooled = [self.maxpool(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        output = self.dropout(self.relu(self.fc(cat)))

        return output


class CNN_LSTM_Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, out_dim, n_filters, filter_sizes, n_layers, dropout, pad_idx):
        super().__init__()

        self.out_dim = out_dim
        
        self.hid_dim = hid_dim
        
        self.n_layers = n_layers
        
        self.n_filters = n_filters
        
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = emb_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        
        self.rnn = nn.LSTM(n_filters, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
               
        #self.relu = leaky_relu
        self.relu = selu
        
        self.fc = nn.Linear(n_layers*len(filter_sizes)*hid_dim, out_dim)

    def forward(self, src, batch_size):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]
        
        embedded = embedded.permute(1, 2, 0)
        #embedded = [batch size, emb dim, src len]
        
        conved = [self.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch_size, n_filters, src len - filter_sizes[n] +1 ]
        
        rev_conved = [conv.permute(2,0,1) for conv in conved]
        #rev_conved_n = [src len - filter_size[n]+1, batch size, n_filters]
        
        rnned = []
        for conv in rev_conved:
            output, (hidden, cell) = self.rnn(conv)
            #hidden = [n layers * n directions, batch size, hid dim]
        
            hidden = hidden.permute(1,0,2)
            hidden = torch.reshape(hidden,[batch_size,self.n_layers*self.hid_dim])
            rnned.append(hidden)
        #rnned_n = [batch size, n_layers*hid dim]
        
        cat = self.dropout(torch.cat(rnned, dim = 1))
        
        output = self.dropout(self.relu(self.fc(cat)))

        return output


class Seq2Func(nn.Module):
    def __init__(self, protein_encoder, smiles_encoder, hid_dim, out_dim, dropout, device):
        super().__init__()
        
        self.protein_encoder = protein_encoder
        
        self.smiles_encoder = smiles_encoder
        
        self.device = device
        
        self.fc1 = nn.Linear(protein_encoder.out_dim+smiles_encoder.out_dim, hid_dim)
        
        self.fc2 = nn.Linear(hid_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.relu = leaky_relu
        
    def forward(self, protein_src, smiles_src):
        
        #Get protein encoder output
        protein_output = self.protein_encoder(protein_src, protein_src.shape[1]) 
        #protein_output = [batch size, protein out_dim]
        
        #Get smiles encoder output
        smiles_output = self.smiles_encoder(smiles_src, smiles_src.shape[1])
        #smiles_output = [batch size, smiles out_dim]
        
        ls_output = torch.cat((protein_output,smiles_output),1)
        #ls_output = [batch size, protein out_dim + smiles out_dim]
        
        o1 = self.dropout(self.relu(self.fc1(ls_output)))
        #o1 = [batch size, hid_dim]
        
        final_output = self.relu(self.fc2(o1))
        #final_output = [batch_size, 1]
        
        return final_output


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.05, 0.05)


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        protein_src = batch.Sequence
        smiles_src = batch.canonical_smiles
        trg = batch.pchembl_value
        
        optimizer.zero_grad()
        
        output = model(protein_src, smiles_src).squeeze(1)
        #output = [batch size]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        del protein_src
        del smiles_src
        torch.cuda.empty_cache()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            protein_src = batch.Sequence
            smiles_src = batch.canonical_smiles
            trg = batch.pchembl_value

            output = model(protein_src, smiles_src).squeeze(1) 
            #output = [batch size]
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            
            del protein_src
            del smiles_src
            torch.cuda.empty_cache()
        
    return epoch_loss / len(iterator)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


