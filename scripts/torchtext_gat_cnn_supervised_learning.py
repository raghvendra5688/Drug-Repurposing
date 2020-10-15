import numpy as np
import pandas as pd
import random
import time
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from torch.nn.functional import relu,leaky_relu
from torch.nn import Linear
from torch.nn import BatchNorm1d
import networkx as nx
from rdkit import Chem
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric import data as DATA
from torch_geometric.data import Data, DataLoader
from math import sqrt
from rdkit.Chem import AllChem
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool
import matplotlib.pyplot as plt
import pickle

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if(mol is None):
        return None
    else:
    
        c_size = mol.GetNumAtoms()
    
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append( feature / sum(feature) )

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
        
        return c_size, features, edge_index


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])


class Net(torch.nn.Module):
    def __init__(self):

        super(Net, self).__init__()

        # SMILES graph branch
        #self.n_output = n_output
        self.conv1 = GATConv(78, 78, heads=2, dropout=0.1)
        self.conv2 = GATConv(78*2, 78*3, dropout=0.1)
        self.conv3 = GATConv(78*3, 78 * 4, dropout=0.1)
        self.fc_g1 = torch.nn.Linear(78*4, 1024)
        self.bn2 = BatchNorm1d(1024)
        self.fc_g2 = Linear(1024, 128)
       
        
        ## Protein Sequences
        n_filters = 512
        self.embedding_xt = nn.Embedding(21 + 1, 128)
        self.conv_xt_1 = nn.Conv1d(in_channels=2000, out_channels=n_filters, kernel_size=3)
        self.conv_xt_2 = nn.Conv1d(in_channels= 512, out_channels= 128, kernel_size=5)
        self.conv_xt_3 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=8) 
        #self.conv_xt_4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3)
        self.fc1_xt1 = nn.Linear(32*11, 256)
        self.bn3 = BatchNorm1d(256)
        self.fc1_xt2 = nn.Linear(256,128)


        self.fc12 = nn.Linear(2*128, 384)
        self.fc22 = nn.Linear(384, 128)
        self.out3 = nn.Linear(128, 1)
        

    def forward(self, data):
        # get graph input
        self.relu = leaky_relu
        flat = nn.Flatten()
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = F.relu(self.fc_g1(x))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.fc_g2(x))
        
        
        
        # Proteins
        embedded_xt = self.embedding_xt(target)
        conv_xt = F.relu(F.max_pool1d(self.conv_xt_1(embedded_xt),2))
        conv_xt = F.relu(F.max_pool1d(self.conv_xt_2(conv_xt),2))
        #conv_xt = self.conv_xt_1(embedded_xt)
        #conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = F.relu(F.max_pool1d(self.conv_xt_3(conv_xt), 2))
        #print("Shape of Conv layer: ", conv_xt.shape)
        #xt = flat(conv_xt)
        xt = conv_xt.view(-1, 32*11)
        #print("Flatten XT shape: ", xt.shape)
        xt = F.relu(self.fc1_xt1(xt))
        xt = F.dropout(xt, p=0.25)
        xt = F.relu(self.fc1_xt2(xt))
        xt = F.dropout(xt, p=0.25)
        
        xc = torch.cat((x, xt), 1)
        #print(xc.shape)
        #print(len(xc[0]))
        # add some dense layers
        xc = F.relu(self.fc12(xc))
        #xc = F.relu(xc)
        xc = F.dropout(xc, p=0.2)
        xc = F.relu(self.fc22(xc))
        xc = F.dropout(xc, p=0.2)
        out = self.out3(xc)
        return out

loss_fn = nn.MSELoss()
best_mse = 1000
calculated_mse = 1000

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse


# ################################## Test Mode #####################################

df = pd.read_csv("../data/Test_Compound_Viral_interactions_for_Supervised_Learning.csv")
protein_seqs = df['Sequence'].values.tolist()
seq_voc_dic = "ACDEFGHIKLMNPQRSTVWXY"
seq_dict = {voc:idx for idx,voc in enumerate(seq_voc_dic)}
seq_dict_len = len(seq_dict)
max_seq_len = 2000

def seq_dict_fun(prot):
    x = np.zeros(max_seq_len)
    x += 21
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  

for i in range(len(protein_seqs)):
    for j in range(len(protein_seqs[i])):
        if(protein_seqs[i][j] in seq_voc_dic):
            continue
        else:
            protein_seqs[i][j] = 'X'


PS = [seq_dict_fun(k) for k in protein_seqs]
pt = []
for i in range(len(PS)):
    pt.append(PS[i])
protein_inputs = np.array(pt)

for i in range(len(protein_seqs)):
    for j in range(len(protein_seqs[i])):
        if(protein_seqs[i][j] in seq_voc_dic):
            continue
        else:
            protein_seqs[i][j] = 'X'



smiles = df['canonical_smiles'].values.tolist()
y = df['pchembl_value'].values.tolist()
uniprot = df['uniprot_accession']
inchi = df['standard_inchi_key']


smile_graph = {}
none_smiles = []
got_g = []

for smile in smiles:
    g = smile_to_graph(smile)
    if(g is None):
        print(smile)
        none_smiles.append(smile)
    else:
        got_g.append(smile)
        smile_graph[smile] = g


smile_graphs = smile_graph

data_features = []
data_edges = []
data_c_size = []
labels = []
data_list = []
for i in range(len(smiles)):
    if(smiles[i] == 'Nc1ccc([S+]2(=O)Nc3nccc[n+]3['):
        print(i)
    else:
        c_size, features, edge_index = smile_graph[smiles[i]]
        data_features.append(features)
        data_edges.append(edge_index)
        data_c_size.append(c_size)
        labels = y[i]
        target = protein_inputs[i]

        GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
        GCNData.target = torch.LongTensor([target])

        GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
        data_list.append(GCNData)

test_X = data_list
test_loader = DataLoader(test_X, batch_size=1, shuffle=False, drop_last=False)

loss_fn = nn.MSELoss()
best_mse = 1000
calculated_mse = 1000


def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = torch.device('cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model.load_state_dict(torch.load('../models/gat_cnn_models/GAT_CNN_2000_3_pooling_checkpoint.pt',map_location=device))#['state_dict'])

model.eval()
total_preds = torch.Tensor()
total_labels = torch.Tensor()

print("Predicting...")
total_pred = []
total_labels = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        total_labels.append(data.y.cpu().data.numpy().tolist()[0])
        total_pred.append(output.cpu().data.numpy()[0].tolist()[0])
t = np.array(total_labels)
p = np.array(total_pred)
pred1 = mse(t,p)
print("Saving results...")
scores = []
for i in range(len(p)):
    tk = []
    tk.append(uniprot[i])
    tk.append(inchi[i])
    tk.append(p[i])
    tk.append(t[i])
    scores.append(tk)

f1 = pd.DataFrame(scores)
f1.columns =['uniprot_accession', 'standard_inchi_key', 'predictions', 'labels']
f1.to_csv("GAT_CNN_Test_set.csv", index=False)
print("Results saved...")


