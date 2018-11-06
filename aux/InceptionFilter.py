import torch.nn as nn
import torch
import time
import numpy as np


class InceptionFilter(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()
        self.Activation = activation
        self.cnn1 = nn.Sequential(
                                nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=True),                                 self.Activation()
                                )
        self.cnn2 = nn.Sequential(
                                nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, bias=True),                                 self.Activation(),
                                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),                                 self.Activation()
                                )
        self.cnn3 = nn.Sequential(
                                nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, bias=True),                                 self.Activation(),
                                nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2, bias=True),                                 self.Activation()
                                )
        self.cnn4 = nn.Sequential(
                                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),                                 self.Activation(),
                                nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=True),                                 self.Activation()
                                )
        
        self.layers = [self.cnn1, self.cnn2, self.cnn3, self.cnn4]
        self.initializeWeights()




    def forward(self, image):
        x1 = self.cnn1(image)
        x2 = self.cnn2(image)
        x3 = self.cnn3(image)
        x4 = self.cnn4(image)
        return torch.cat([x1,x2,x3,x4],dim=1)


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




    def save(self, fname="Inception_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
