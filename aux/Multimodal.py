import torch.nn as nn
import torch
import time
import numpy as np
from VisionModality import VisionModality
from ScentModality import ScentModality
import cv2

class Multimodal(nn.Module):
    def __init__(self, state_size, action_space, activation=nn.ReLU):
        super().__init__()
        self.Activation = activation
        self.vision = VisionModality(self.Activation)
        self.scent = ScentModality(self.Activation)
        self.fc1  = nn.Sequential(
                                nn.Linear(state_size, 2*state_size, bias=True),                                 self.Activation(),
                                nn.Linear(2*state_size, 4*state_size, bias=True),                                self.Activation()
                                )
        self.lstm = nn.LSTM(input_size=4*state_size+128+64, hidden_size=256, num_layers=3, dropout=0, bidirectional=True)
        self.fc2  = nn.Sequential(
                                nn.Linear(512, 128, bias=True),                                 self.Activation(),
                                nn.Linear(128, 32, bias=True),                                self.Activation()
                                )
        self.policy = nn.Linear(32, action_space, bias=True)
        self.value = nn.Linear(32, 1, bias=True)
        self.layers = [self.vision, self.scent, self.fc1, self.lstm, self.fc2, self.policy, self.value]
        self.initializeWeights()




    def forward(self, image, scent, state, hidden_vision=None, hidden_scent=None, hidden_state=None):
        batch_size = image.size(0)
        sequence_length = image.size(1)
        image,hidden_vision = self.vision.forward(image,hidden_vision)
        scent,hidden_scent = self.scent.forward(scent,hidden_scent)
        state = self.fc1.forward(state.view(batch_size*sequence_length,-1)).view(batch_size,sequence_length,-1)
        embedding = torch.cat([image,scent,state],dim=-1).permute(1,0,2)
        x, hidden_state = self.lstm(embedding, hidden_state)
        x = self.fc2(x.permute(1,0,2).view(batch_size*sequence_length,-1))
        value = self.value(x).view(batch_size,sequence_length,-1)
        policy = self.policy(x).view(batch_size,sequence_length,-1)
        return value, policy


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
        for name,parameter in self.named_parameters():
            cv2.imwrite(name,parameter)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
