import torch.nn as nn
import torch
import time
import numpy as np
from Multimodal import Multimodal
from PixelControl import PixelControl
from FeatureControl import FeatureControl
from RewardPrediction import RewardPrediction
import cv2

class AuxNetwork(nn.Module):
    def __init__(self, state_size, action_space=3, num_input_to_fc=3872, activation=nn.ReLU):
        super().__init__()
        self.Activation = activation
        self.Multimodal = Multimodal(num_input_to_fc, state_size, action_space, activation)        
        self.PixelControl = PixelControl(state_size, action_space, activation)
        self.FeatureControl = FeatureControl(state_size, action_space, activation)
        self.RewardPrediction = RewardPrediction(num_input_to_fc, self.Multimodal.vision.encoders, activation)
        self.layers = [self.Multimodal, self.PixelControl, self.FeatureControl, self.RewardPrediction]
        self.initializeWeights()




    def forward(self, image, scent, state, hidden_vision=None, hidden_scent=None, hidden_state=None):
        batch_size = image.size(0)
        sequence_length = image.size(1)
        vision_lstm_ouput, value, policy = self.Multimodal.forward(image, scent, state, hidden_vision=None, hidden_scent=None, hidden_state=None)
        pc_action_value = self.PixelControl.forward(vision_lstm_ouput)
        fc_action_value = self.FeatureControl.forward(vision_lstm_ouput)
        return value, policy, pc_action_value, fc_action_value



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




    def save(self, fname="Aux_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        for name,parameter in self.named_parameters():
            try:
                cv2.imwrite("{}.png".format(name),parameter.numpy())
            except:
                pass
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
