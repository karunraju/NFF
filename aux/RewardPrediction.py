import torch.nn as nn
import torch
import time
import numpy as np

class RewardPrediction(nn.Module):
    def __init__(self, num_input_to_fc,encoders, activation=nn.ReLU):
        super().__init__()
        self.Activation = activation
        self.encoders = encoders
        self.fc1  = nn.Sequential(nn.Linear(num_input_to_fc, 2*num_input_to_fc, bias=True), self.Activation(),
                                  nn.Linear(2*num_input_to_fc, 2*num_input_to_fc, bias=True), self.Activation(),
                                  nn.Linear(2*num_input_to_fc, 3, bias=True))
        self.layers = [self.fc1]
        self.initializeWeights()




    def forward(self, image):  # B x concat x 3 x 11 x 11
        batch_size = image.size(0)
        concat_size = image.size(1)
        x = image.view(-1,image.size(-3),image.size(-2),image.size(-1))
        for enc in self.encoders:
            with torch.no_grad():
                x = enc(x)
        x = self.fc1(x.view(batch_size,-1))
        return nn.functional.Softmax(x,dim=-1)


    def initializeWeights(self, function=nn.init.xavier_normal_):
        for layer in self.layers:
            try:
                layer.initializeWeights()
            except:
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




    def save(self, fname="Reward_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
