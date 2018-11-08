import torch.nn as nn
import torch
import time
import numpy as np
from aux.InceptionFilter import InceptionFilter
import hyperparameters as PARAM

class VisionModality(nn.Module):
    def __init__(self, num_input_to_fc, activation=nn.ReLU):
        super().__init__()
        self.Activation = activation
        if PARAM.INCEPTION_FILTER:
          self.cnn1 = InceptionFilter(self.Activation)
          self.cnn2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True), self.Activation())
        else:
          self.cnn1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=True), self.Activation())
          self.cnn2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=True), self.Activation())

        self.fc1  = nn.Sequential(nn.Linear(num_input_to_fc, 2*num_input_to_fc, bias=True), self.Activation(),
                                  nn.Linear(2*num_input_to_fc, 256, bias=True), self.Activation())
        if PARAM.bidirectional:
          self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, dropout=0, bidirectional=True)
          self.fc2  = nn.Sequential(nn.Linear(512, 256, bias=True), self.Activation(),
                                    nn.Linear(256, 128, bias=True), self.Activation())
        else:
          self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, dropout=0)
          self.fc2  = nn.Sequential(nn.Linear(256, 256, bias=True), self.Activation(),
                                    nn.Linear(256, 128, bias=True), self.Activation())
        self.layers = [self.cnn1, self.cnn2, self.fc1, self.lstm, self.fc2]
        self.initializeWeights()
        self.encoders =  [self.cnn1,self.cnn2]


    def forward(self, image, hidden_vision=None):  # B x L x C x 11 x 11
        batch_size = image.size(0)
        sequence_length = image.size(1)
        x = self.cnn1(image.view(-1, image.size(2), image.size(3), image.size(4)))
        x = self.cnn2(x)                    #gives BL x C x H x W
        x = self.fc1(x.view(x.size(0),-1)).view(batch_size,sequence_length,-1).permute(1,0,2)
        lstm_output,hidden_vision = self.lstm(x,hidden_vision)

        x = self.fc2(lstm_output).view(batch_size, sequence_length, -1)
        return lstm_output.permute(1, 0, 2), x, hidden_vision


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


    def save(self, fname="Vision_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
