import torch.nn as nn
import torch
import time
import numpy as np

class PixelControl(nn.Module):
    def __init__(self, state_size, action_space, activation=nn.ReLU):
        super().__init__()
        self.Activation = activation
        self.action_space=action_space
        self.input_linear = nn.Sequential(  nn.Linear(512, 1024),            self.Activation(),
                                            nn.Linear(1024, 2048),            self.Activation()
                                    )
        self.decon = nn.Sequential(         nn.ConvTranspose2d(32, 16, 4, stride=1, padding=0, output_padding=0),   self.Activation(),
                                            nn.ConvTranspose2d(16, 1, 4, stride=1, padding=0, output_padding=0),    self.Activation()
                                    )
        deconv_size=256
        self.ouput_common_linear = nn.Sequential(  nn.Linear(deconv_size, deconv_size//2),                                       self.Activation(),
                                                   nn.Linear(deconv_size//2, deconv_size//4),                                    self.Activation()
                                    )
        self.value_linear = nn.Sequential(  nn.Linear(deconv_size//4, deconv_size//8),                                           self.Activation(),
                                            nn.Linear(deconv_size//8, 1)
                                    )
        self.advantage_linear = nn.Sequential(  nn.Linear(deconv_size//4, deconv_size//8),                                       self.Activation(),
                                                nn.Linear(deconv_size//8, 4)
                                    )

        self.layers = [self.input_linear, self.decon, self.ouput_common_linear, self.value_linear, self.advantage_linear]
        self.initializeWeights()




    def forward(self, lstm_out):            # B x L x 512
        batch_size = lstm_out.size(0)
        sequence_length = lstm_out.size(1)
        x = lstm_out.view(batch_size*sequence_length,-1)
        x = self.input_linear(x).view(batch_size*sequence_length,32,-1)
        x = self.decon(x.view(batch_size*sequence_length,32,x.size(-1)//2,-1))
        x = self.ouput_common_linear(x.view(batch_size*sequence_length,-1))
        value = self.value_linear(x).view(batch_size,sequence_length,-1)
        advantage = self.advantage_linear(x).view(batch_size,sequence_length,-1)
        action_value = value.repeat(1,1,self.action_space) + advantage - advantage.mean(dim=-1)  #Is the mean along batch axis or per state
        return action_value         #B x L x action_space


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




    def save(self, fname="Pixel_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        for name,parameter in self.named_parameters():
            cv2.imwrite(name,parameter)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
