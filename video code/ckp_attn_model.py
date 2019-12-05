import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# config = Config(
#     num_classes = 7,s
#     width = 224,
#     height = 224,
#     num_epochs = 30,
#     batch_size = 32,
#     feat_dim = 11,
#     lr_cent = 0.5,
#     closs_weight = 0.5,
#     ckp = True,
#     device = 'cpu'
# )

device = 'cpu'

class AttentionLayer(nn.Module):
    def __init__(self, input1_size, input2_size):
        super(AttentionLayer, self).__init__()

        self.attention_fclayer = nn.Linear(input2_size, input1_size)
        
    def forward(self, input1, input2):
        
        self.input1 = input1
        self.input2 = input2
        
        self.batch, self.outchannel1, self.h1, self.w1 = input1.shape
        self.batch, self.outchannel2 = input2.shape
        
        input2_rescaled = self.attention_fclayer(self.input2)
        
        input2_bmm = input2_rescaled.view(self.batch,self.outchannel1,1)
        
        compat_scores = torch.zeros((self.batch, self.h1*self.w1))
        compat_scores = compat_scores.to(device)

        for h in range(self.h1):
            for w in range(self.w1):
                input1_bmm = self.input1[:,:,h,w].view(self.batch,1,self.outchannel1)
                compat_scores[:,h*self.w1+w] = torch.bmm(input1_bmm, input2_bmm).squeeze()
        
        normalized_compat_scores = F.softmax(compat_scores, dim=1)
        
        bmm_arg2 = self.input1.view(self.batch,self.outchannel1,self.h1*self.w1,1)
        bmm_argtemp = normalized_compat_scores.view(self.batch,1,self.h1*self.w1).repeat(1,self.outchannel1,1)
        bmm_arg1 = bmm_argtemp.view(self.batch,self.outchannel1,1,self.h1*self.w1)
        
        g_mod = torch.zeros((self.batch, self.outchannel1))
        g_mod = g_mod.to(device)
        
        for b in range(self.batch):
            g_mod[b,:] = torch.bmm(bmm_arg1[b,:,:,:], bmm_arg2[b,:,:,:]).squeeze()
        
        return g_mod
    
class ConvBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
                          nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, stride=stride, padding=(1,1)),
                          nn.BatchNorm2d(C_out),
                          nn.ReLU())
        
    def forward(self, x):
        return self.block(x)

class LinearBlock(nn.Module):
    def __init__(self, insize, outsize):
        super(LinearBlock, self).__init__()
        self.linblock = nn.Sequential(
                          nn.Linear(insize, outsize),
                          nn.BatchNorm1d(outsize),
                          nn.ReLU())
        
    def forward(self, x):
        return self.linblock(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class BaselineModel(nn.Module):
    def __init__(self, num_blocks):
        super(BaselineModel, self).__init__()
        layers = []
        num_classes = 7
        channels = [1, 256, 128, 64] # this needs to be modified according to num_blocks
        linear_size = [64*6*6, 512, 256, 128]
        
        self.conv1 = ConvBlock(C_in=channels[0], C_out=channels[1], kernel_size=3, stride=1)
        self.conv2 = ConvBlock(C_in=channels[1], C_out=channels[1], kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv3 = ConvBlock(C_in=channels[1], C_out=channels[2], kernel_size=3, stride=1)
        self.conv4 = ConvBlock(C_in=channels[2], C_out=channels[2], kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv5 = ConvBlock(C_in=channels[2], C_out=channels[3], kernel_size=3, stride=1)
        self.conv6 = ConvBlock(C_in=channels[3], C_out=channels[3], kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(2)
        
        self.flatten = Flatten()
        
        self.fc1 = LinearBlock(linear_size[0], linear_size[1])
        self.fc2 = LinearBlock(linear_size[1], linear_size[2])
        self.fc3 = LinearBlock(linear_size[2], linear_size[3])
        
        self.attlayer = AttentionLayer(channels[3], linear_size[3])
        
        self.fc4 = nn.Linear(channels[3], 7) #config.num_classes
        
    def forward(self, x):
        
        self.out1 = self.conv1(x)
        self.out2 = self.conv2(self.out1)
        self.out3 = self.pool1(self.out2)
        
        self.out4 = self.conv3(self.out3)
        self.out5 = self.conv4(self.out4)
        self.out6 = self.pool2(self.out5)
        
        self.out7 = self.conv5(self.out6)
        self.out8 = self.conv6(self.out7)
        self.out9 = self.pool3(self.out8)
        
        self.out10 = self.flatten(self.out9)
        
        self.out11 = self.fc1(self.out10)
        self.out12 = self.fc2(self.out11)
        self.out13 = self.fc3(self.out12)
        
        self.attout = self.attlayer(self.out7, self.out13)
        
        self.out = self.fc4(self.attout)
        
        return self.out


