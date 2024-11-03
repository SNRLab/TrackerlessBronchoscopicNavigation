

from layers import *

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    
class FCN(nn.Module):
    
    def __init__(self, output_size = 1):
        super(FCN, self).__init__()

        self.flatten = Flatten()
        self.ouput_size = output_size
        self.dropout = nn.Dropout(0.2).to('cuda')
        self.nonlinear = nn.LeakyReLU(negative_slope=0.01).to('cuda')
        
        self.normalization = {}
        self.conv = {}
        self.linear = {}
        
        base_channels = 512
        self.normalization1 = nn.InstanceNorm2d(base_channels//2**(1)).to('cuda')
        self.normalization2 = nn.InstanceNorm2d(base_channels//2**(2)).to('cuda')
        self.normalization3 = nn.InstanceNorm2d(base_channels//2**(3)).to('cuda')
        
        self.conv1 = nn.Conv2d(base_channels//2**(0), base_channels//2**(1), 3, padding = 'valid').to('cuda')
        self.conv2 = nn.Conv2d(base_channels//2**(1), base_channels//2**(2), 3, padding = 'valid').to('cuda')
        self.conv3 = nn.Conv2d(base_channels//2**(2), base_channels//2**(3), 3, padding = 'valid').to('cuda')
        
        # for i in range(1, 4): 
        #     self.normalization[i-1] = nn.InstanceNorm2d(base_channels//2**(i)).to('cuda')
        #     self.conv[i-1] = nn.Conv2d(base_channels//2**(i-1), base_channels//2**(i), 3, padding = 'valid').to('cuda')
            # self.normalization.append(nn.InstanceNorm2d(base_channels//2**(i)).to('cuda'))
            # self.conv.append(nn.Conv2d(base_channels//2**(i-1), base_channels//2**(i), 3, padding = 'valid').to('cuda'))
        
        ch = 64*6*6
        # final_ch_size = 0 
        self.linear1 = nn.Linear(ch//2**(0), ch//2**(1)).to('cuda')
        self.linear2 = nn.Linear(ch//2**(1), ch//2**(2)).to('cuda')
        self.linear3 = nn.Linear(ch//2**(2), output_size).to('cuda')
        # for i in range(1,3):
        #     self.linear[i-1] = nn.Linear(ch//2**(i-1), ch//2**(i)).to('cuda')
        #     final_ch_size = ch//2**(i)
        # #    self.linear.append(nn.Linear(ch//2**(i-1), ch//2**(i)).to('cuda'))
        # #    final_ch_size = ch//2**(i)
        
        # # the last one
        # self.linear[3] = nn.Linear(final_ch_size, output_size).to('cuda')
        # self.linear.append(nn.Linear(final_ch_size, output_size).to('cuda'))
        
        self.relu = nn.ReLU().to('cuda')
        
        self.sigmoid = nn.Sigmoid().to('cuda')
        # self.initializeWeights()
        
        self.model = nn.Sequential(
            self.conv1,
            self.normalization1, 
            self.nonlinear, 
            self.dropout,
            
            self.conv2,
            self.normalization2, 
            self.nonlinear, 
            self.dropout, 
            
            self.conv3,
            self.normalization3, 
            self.nonlinear, 
            self.dropout, 
            
            self.flatten, 
            self.linear1,
            self.nonlinear,
            
            self.linear2,
            self.nonlinear,
            self.linear3, 
            self.sigmoid
        )
            

    def forward(self, x):
        return self.model(x)




class FCN_free_mask(nn.Module):
    
    def __init__(self, output_size = 3):
        super(FCN_free_mask, self).__init__()

        base_channels = 512
        
        self.up1 = (Up(base_channels, base_channels//2, bilinear = True))# self.base_filter*16, self.base_filter*8
        self.up2 = (Up(base_channels//2, base_channels//4, bilinear = True))# self.base_filter*8, self.base_filter*4
        self.up3 = (Up(base_channels//4, base_channels//8, bilinear = True))# self.base_filter*4, self.base_filter*2
        self.up4 = (Up(base_channels//8, base_channels//16, bilinear = True)) # self.base_filter*2, self.base_filter*1
        self.outc = (OutConv(base_channels//16, output_size))
        
        self.sigmoid = nn.Sigmoid().to('cuda')
      
        self.model = nn.Sequential(
            self.up1,
            self.up2, 
            self.up3, 
            self.up4,
            self.sigmoid
        )
            

    def forward(self, x):
        return self.model(x)
    



