import torch.nn as nn
import torch
import time
import numpy as np
from aux.Multimodal import Multimodal
from aux.PixelControl import PixelControl
from aux.FeatureControl import FeatureControl
from aux.RewardPrediction import RewardPrediction
try:
  import cv2
except ImportError:
  pass

class AuxNetwork(nn.Module):
    def __init__(self, state_size, seq_len=1, action_space=3, num_input_to_fc=3872, activation=nn.ReLU):
        super(AuxNetwork, self).__init__()
        self.Activation = activation
        self.Multimodal = Multimodal(num_input_to_fc, state_size, action_space, activation, seq_len=seq_len)
        self.PixelControl = PixelControl(state_size, action_space, activation)
        self.FeatureControl = FeatureControl(state_size, action_space, activation)
        self.RewardPrediction = RewardPrediction(num_input_to_fc, self.Multimodal.vision.encoders, activation)
        self.layers = [self.Multimodal, self.PixelControl, self.FeatureControl, self.RewardPrediction]
        self.initializeWeights()

    def forward(self, image, scent, state, hidden_vision=None, hidden_scent=None, hidden_state=None):
        vision_lstm_ouput, value, policy = self.Multimodal.forward(image, scent, state, hidden_vision=None, hidden_scent=None, hidden_state=None)
        return value, policy


    def predict_rewards(self, image):
        return self.RewardPrediction(image)


    def pixel_control(self, image, hidden_vision=None):
        vision_lstm_ouput = self.Multimodal.vision_lstm_output(image, hidden_vision=None)
        pc_action_value = self.PixelControl.forward(vision_lstm_ouput)
        return pc_action_value


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
        """
        for name,parameter in self.named_parameters():
            try:
                cv2.imwrite("{}.png".format(name),parameter.clone().detach().cpu().numpy())
            except:
                pass
        """
        return fname
    def load(self, fname):
        self.load_state_dict(torch.load(fname))
