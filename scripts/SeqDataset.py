import torch
import torch.utils

class SeqDataset(torch.utils.data.Dataset):

    def __init__(self, sm_list):
        self.smiles = sm_list
        

    def __getitem__(self, num):
        return self.smiles[num]
    
    def __len__(self):
        return len(self.smiles)
