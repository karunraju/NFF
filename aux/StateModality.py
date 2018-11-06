import torch.nn as nn
import torch
import time
import numpy as np

class ScentModality(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()
        self.Activation = activation
        self.fc1  = nn.Sequential(
                                nn.Linear(3, 4*3, bias=True),                                 self.Activation()
                                nn.Linear(4*3, 8*3, bias=True),                                self.Activation()
                                )
        self.lstm = nn.LSTM(input_size=8*3, hidden_size=128, num_layers=3, dropout=0, bidirectional=True)
        self.fc2  = nn.Sequential(
                                nn.Linear(256, 128, bias=True),                                 self.Activation()
                                nn.Linear(128, 64, bias=True),                                self.Activation()
                                )
        self.layers = [self.fc1, self.lstm, self.fc2]
        self.initializeWeights()




    def forward(self, image):
        x = self.fc1(x)
        x = self.lstm(x)
        x = self.fc2(x)
        return x


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




    def save(self, fname="Vision_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
