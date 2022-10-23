import re
import nltk
import numpy as np
# nltk.download('punkt')
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init


class CBOW(nn.Module):
    
    def __init__(
        self, 
        embedding_dim:int, 
        input_dim:int, 
        window_size:int,
        
        enc_dim:list,
        dec_dim:list,
        ls_dim:list,
        
        enc_dropout:list,
        dec_dropout:list,
        ls_dropout:list,
        
        enc_act=None,
        ls_act=None,
        dec_act=None
    ):
        super(CBOW, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        self.ls_dim = ls_dim
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        
        self.enc_dropout = enc_dropout
        self.ls_dropout = ls_dropout
        self.dec_dropout = dec_dropout
        
        self.enc_act = enc_act
        self.ls_act = ls_act
        self.dec_act = dec_act
                
        # Encoder
        self.Encoder = nn.Sequential()
        self.enc_dim = [self.input_dim * (self.window_size - 1)] + self.enc_dim
        if enc_act is None:
            self.enc_act = [nn.ReLU() for _ in self.enc_dim]
        for idx in range(len(self.enc_dim) - 1):
            # FC
            module = nn.Linear(self.enc_dim[idx], self.enc_dim[idx + 1])
            init.xavier_uniform_(module.weight)
            self.Encoder.add_module(name="Encoder_(L_%s)"%(idx), module=module)
            # Activation
            self.Encoder.add_module(name="Encoder_(Act_%s)"%(idx), module=self.enc_act[idx])
            # Dropout
            module = nn.Dropout(self.enc_dropout)
            self.Encoder.add_module(name="Encoder_(Dropout_%s)"%(idx), module=module)    

        # Latent Space
        self.LatentSpace = nn.Sequential()
        self.ls_dim = [self.enc_dim[-1]] + self.ls_dim + [self.embedding_dim]
        if ls_act is None:
            self.ls_act = [nn.ReLU() for _ in self.ls_dim]
        for idx in range(len(self.ls_dim) - 1):
            # FC
            module = nn.Linear(self.ls_dim[idx], self.ls_dim[idx + 1])
            init.xavier_uniform_(module.weight)
            self.LatentSpace.add_module(name="Lat_Sp_(L_%s)"%(idx), module=module)
            # Activation
            self.LatentSpace.add_module(name="Lat_Sp_(Act_%s)"%(idx), module=self.ls_act[idx])
            # Dropout
            module = nn.Dropout(self.ls_dropout)
            self.LatentSpace.add_module(name="Lat_Sp_(Dropout_%s)"%(idx), module=module)    

        # Decoder
        self.Decoder = nn.Sequential()
        self.dec_dim = [self.embedding_dim] + self.dec_dim + [self.input_dim]
        if dec_act is None:
            self.dec_act = [nn.ReLU() for _ in self.dec_dim]
        for idx in range(len(self.dec_dim)- 1):
            # FC
            module = nn.Linear(self.dec_dim[idx], self.dec_dim[idx + 1])
            init.xavier_uniform_(module.weight)
            self.Decoder.add_module(name="Decoder_(L_%s)"%(idx), module=module)            
            # Activation
            self.Decoder.add_module(name="Decoder_(Act_%s)"%(idx), module=self.dec_act[idx])
            # Dropout
            module = nn.Dropout(self.dec_dropout)
            self.Decoder.add_module(name="Decoder_(Dropout_%s)"%(idx), module=module)    

    def forward(self, inputs):
        enc_out = torch.cat([self.Encoder(inp) for inp in inputs])
        lat_sp_out = self.LatentSpace(enc_out)
        dec_out = self.Decoder(lat_sp_out)
        dec_out = F.log_softmax(dec_out, dim=0)
        return dec_out