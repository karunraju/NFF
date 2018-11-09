import torch.nn as nn
import torch
import time
import numpy as np
from aux.VisionModality import VisionModality
from aux.ScentModality import ScentModality
import hyperparameters as PARAM
try:
  import cv2
except ImportError:
  pass


class Multimodal(nn.Module):
    def __init__(self, num_input_to_fc, state_size, action_space, activation=nn.ReLU, seq_len=1):
        super().__init__()
        self.Activation = activation
        self.vision = VisionModality(num_input_to_fc, self.Activation)
        if PARAM.SCENT_MODALITY:
          self.scent = ScentModality(self.Activation)
        else:
          self.scent = nn.Sequential(nn.Linear(3, 32, bias=True), self.Activation())
        self.fc1  = nn.Sequential(nn.Linear(state_size, 2*state_size, bias=True), self.Activation(),
                                  nn.Linear(2*state_size, 4*state_size, bias=True), self.Activation())
        if PARAM.bidirectional:
          self.lstm = nn.LSTM(input_size=4*state_size+128+32, hidden_size=256, num_layers=3, dropout=0, bidirectional=True)
          self.fc2  = nn.Sequential(nn.Linear(512, 128, bias=True), self.Activation(),
                                    nn.Linear(128, 32, bias=True), self.Activation())
        else:
          self.lstm = nn.LSTM(input_size=4*state_size+128+32, hidden_size=256, num_layers=3, dropout=0)
          self.fc2  = nn.Sequential(nn.Linear(256, 128, bias=True), self.Activation(),
                                    nn.Linear(128, 32, bias=True), self.Activation())
        if PARAM.MLP_ACROSS_TIME:
            self.policy = nn.Linear(32*seq_len, action_space, bias=True)
            self.value = nn.Linear(32*seq_len, 1, bias=True)
        else:
            self.policy = nn.Linear(32, action_space, bias=True)
            self.value = nn.Linear(32, 1, bias=True)
        self.layers = [self.vision, self.scent, self.fc1, self.lstm, self.fc2, self.policy, self.value]
        self.initializeWeights()


    def forward(self, image, scent, state, hidden_vision=None, hidden_scent=None, hidden_state=None):
        batch_size = image.size(0)
        sequence_length = image.size(1)
        vision_lstm_output, image, hidden_vision = self.vision.forward(image,hidden_vision)
        if PARAM.SCENT_MODALITY:
          scent, hidden_scent = self.scent.forward(scent, hidden_scent)
        else:
          scent = self.scent(scent)
        state = self.fc1.forward(state.view(batch_size*sequence_length,-1)).view(batch_size,sequence_length,-1)
        embedding = torch.cat([image,scent,state],dim=-1).permute(1,0,2)
        lstm_ouput, hidden_state = self.lstm(embedding, hidden_state)
        if PARAM.MLP_ACROSS_TIME:
            x = self.fc2(lstm_ouput).view(batch_size*sequence_length, -1)
            value = self.value(x.view(batch_size,-1))
            policy = self.policy(x.view(batch_size,-1))
            return vision_lstm_output, value, nn.functional.softmax(policy,dim=-1)
        else:
            lstm_ouput = lstm_ouput.permute(1,0,2).view(batch_size*sequence_length,-1)
            x = self.fc2(lstm_ouput)
            value = self.value(x).view(batch_size,sequence_length,-1)
            policy = self.policy(x).view(batch_size,sequence_length,-1)
            return vision_lstm_output, value[:,-1,:].squeeze(1), nn.functional.softmax(policy,dim=-1)[:,-1,:].squeeze(1)


    def vision_lstm_output(self, image, hidden_vision=None):
        vision_lstm_output, image, hidden_vision = self.vision.forward(image, hidden_vision)
        return vision_lstm_output


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


    def save(self, fname="Multi_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        for name,parameter in self.named_parameters():
            try:
                cv2.imwrite("{}.png".format(name),parameter.numpy())
            except:
                pass
        return fname


    def load(self, fname):
        self.load_state_dict(torch.load(fname))
