
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size=3, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), bias=True,
                 radix=2, reduction_factor=4,norm_layer=nn.BatchNorm2d):
        super(SplAtConv2d, self).__init__()

        padding = _pair(padding)
        #inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.inchannels = in_channels
        self.radix = radix

        self.channels = channels
      
        self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                            groups=radix, bias=bias)
        
        self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.conv1 = Conv2d(channels, self.inchannels, 1, groups=1)
        
        self.bn1 = norm_layer(self.inchannels)
        self.conv2 = Conv2d(self.inchannels, channels*radix, 1, groups=1)

        self.rsoftmax = rSoftMax(radix, 1)

    def forward(self, x):
        x = self.conv(x)       
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        
        splited = torch.split(x, rchannel//self.radix, dim=1)#channels
        gap = sum(splited) 
        
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.conv1(gap)


        #gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.conv2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        attens = torch.split(atten, rchannel//self.radix, dim=1)
        out = sum([att*split for (att, split) in zip(attens, splited)])#out is channels
        
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, stride=1, dilation=1,radix=2, cardinality=1, bottleneck_width=64,):
        super(Bottleneck, self).__init__()
        #group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.in_channels = in_channels

        self.channels = channels
        self.conv1 = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SplAtConv2d(
                self.channels, self.channels, kernel_size=3,
                stride=stride, padding=dilation, bias=False)
        self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
        #outchannel = max(64,self.channels*2)
        self.conv3 = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channels)  


    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)

        return out
class UpBottleneck(nn.Module):
    def __init__(self, in_channels, channels, stride=1, dilation=1,radix=2, cardinality=1, bottleneck_width=64,):
        super(UpBottleneck, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.in_channels = in_channels

        self.channels = channels
        self.conv1 = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SplAtConv2d(
                self.channels, self.channels, kernel_size=3,
                stride=stride, padding=dilation, bias=False)
        #self.avd_layer = nn.AvgPool2d(3, stride, padding=1)#去除池化层

        #outchannel = max(64,self.channels*2)
        self.conv3 = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channels)    

    def forward(self,x):
        x = self.up(x)
        residual = self.conv1(x)
        residual = self.bn1(residual)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return x
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):

        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
class ResUNeSt(nn.Module):
    def __init__(self,in_channels=1):
        super(ResUNeSt, self).__init__()
        #conv_layer = nn.Conv2d
        #self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.bn0 = nn.BatchNorm2d(1)#参数是输出的通道数
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Conv1 = Bottleneck(in_channels=in_channels, channels=64)
        self.Conv2 = Bottleneck(in_channels=64, channels=128)
        self.Conv3 = Bottleneck(in_channels=128, channels=256)
        self.Conv4 = Bottleneck(in_channels=256, channels=512)
        self.Conv5 = Bottleneck(in_channels=512, channels=1024)
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
  
        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = Bottleneck(in_channels=1024,channels=512)
        self.Up4 = up_conv(ch_in=512,ch_out=256)
    
        self.Up_conv4 = Bottleneck(in_channels=512,channels=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)  
        self.Up_conv3 = Bottleneck(in_channels=256,channels=128)
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = Bottleneck(in_channels=128,channels=64)
        self.Conv_1x1 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x1 = self.Conv1(x)#64 256 256
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)#128 128 128

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)#256 64 64
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)#512 32 32
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)#1024 16 16

        d5 = self.Up5(x5)#32,32,512
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)#256,64,64
        d4 = torch.cat((x3,d4),dim=1)#512,64,64
        d4 = self.Up_conv4(d4)#256,64,64

        d3 = self.Up3(d4)#128,128,128
        d3 = torch.cat((x2,d3),dim=1)#256,128,128
        d3 = self.Up_conv3(d3)#128,128,128

        d2 = self.Up2(d3)#64,256,256
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)#64,256,256

        d1 = self.Conv_1x1(d2)#1,256,256

        #d1 = self.relu(d1)

        return d1
if __name__ =="__main__":
    from torchsummary import summary

  

    net = ResUNeSt()
    print(summary(net,(1,256,256)))
