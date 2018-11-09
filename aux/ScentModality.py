import torch.nn as nn
import torch
import time
import numpy as np
import hyperparameters as PARAM

class ScentModality(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()
        self.Activation = activation
        self.fc1  = nn.Sequential(nn.Linear(3, 4*3, bias=True), self.Activation(),
                                  nn.Linear(4*3, 8*3, bias=True), self.Activation())
        if PARAM.bidirectional:
          self.lstm = nn.LSTM(input_size=8*3, hidden_size=64, num_layers=3, dropout=0, bidirectional=True)
        else:
          self.lstm = nn.LSTM(input_size=8*3, hidden_size=128, num_layers=3, dropout=0)
        self.fc2  = nn.Sequential(nn.Linear(128, 64, bias=True), self.Activation(),
                                  nn.Linear(64, 32, bias=True), self.Activation())
        self.layers = [self.fc1, self.lstm, self.fc2]
        self.initializeWeights()


    def forward(self, scent, hidden_scent=None):    #B x L x 3
        batch_size = scent.size(0)
        sequence_length = scent.size(1)
        x = self.fc1(scent.view(batch_size*sequence_length,-1)).view(batch_size,sequence_length,-1).permute(1,0,2)
        x, hidden_scent = self.lstm(x,hidden_scent)
        if PARAM.MLP_ACROSS_TIME:
            x = self.fc2(x).view(batch_size, sequence_length, -1)
        else:
            x = self.fc2(x.permute(1,0,2).view(batch_size*sequence_length,-1)).view(batch_size,sequence_length,-1)
        return x, hidden_scent


    def initializeWeights(self, function=nn.init.xavier_normal_):
        for layer in self.layers:
            try:
                function(layer.weight.data)
                nn.init.constant_(layer.bias.data, 0)
            except:
                try:
                    for l in layer:
                        try:
                            function(l.weight.data)
                            nn.init.constant_(l.bias.data, 0)
                        except:
                            pass
                except:
                    pass


    def save(self, fname="Scent_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
