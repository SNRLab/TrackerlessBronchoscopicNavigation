from __future__ import absolute_import, division, print_function

from layers import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.base_filter = 32
        self.inc = (DoubleConv(n_channels, self.base_filter))
        self.down1 = (Down(self.base_filter, self.base_filter*2)) # self.base_filter, self.base_filter*2
        self.down2 = (Down(self.base_filter*2, self.base_filter*4))# self.base_filter*2, self.base_filter*4
        self.down3 = (Down(self.base_filter*4, self.base_filter*8))# self.base_filter*4, # self.base_filter*8
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.base_filter*8, self.base_filter*16 // factor)) # self.base_filter*8, self.base_filter*16
        self.up1 = (Up(self.base_filter*16, self.base_filter*8 // factor, bilinear))# self.base_filter*16, self.base_filter*8
        self.up2 = (Up(self.base_filter*8, self.base_filter*4 // factor, bilinear))# self.base_filter*8, self.base_filter*4
        self.up3 = (Up(self.base_filter*4, self.base_filter*2// factor, bilinear))# self.base_filter*4, self.base_filter*2
        self.up4 = (Up(self.base_filter*2, self.base_filter*1, bilinear)) # self.base_filter*2, self.base_filter*1
        self.outc = (OutConv(self.base_filter, n_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        output = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        output.append(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        output.append(self.sigmoid(logits))
        return output

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
        
class UNet_instanceNorm(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_instanceNorm, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.base_filter = 32
        self.inc = (DoubleConvIN(n_channels, self.base_filter))
        self.down1 = (DownIN(self.base_filter, self.base_filter*2)) # self.base_filter, self.base_filter*2
        self.down2 = (DownIN(self.base_filter*2, self.base_filter*4))# self.base_filter*2, self.base_filter*4
        self.down3 = (DownIN(self.base_filter*4, self.base_filter*8))# self.base_filter*4, # self.base_filter*8
        factor = 2 if bilinear else 1
        self.down4 = (DownIN(self.base_filter*8, self.base_filter*16 // factor)) # self.base_filter*8, self.base_filter*16
        self.up1 = (UpIN(self.base_filter*16, self.base_filter*8 // factor, bilinear))# self.base_filter*16, self.base_filter*8
        self.up2 = (UpIN(self.base_filter*8, self.base_filter*4 // factor, bilinear))# self.base_filter*8, self.base_filter*4
        self.up3 = (UpIN(self.base_filter*4, self.base_filter*2// factor, bilinear))# self.base_filter*4, self.base_filter*2
        self.up4 = (UpIN(self.base_filter*2, self.base_filter*1, bilinear)) # self.base_filter*2, self.base_filter*1
        self.outc = (OutConv(self.base_filter, n_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        output = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        output.append(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        output.append(self.sigmoid(logits))
        return output

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)