from __future__ import absolute_import, division, print_function

from layers import *

class GaussianLayer(nn.Module):
    def __init__(self, input_size, num_of_gaussians = 1):
        super(GaussianLayer, self).__init__()
        
        self.input_size = input_size
        self.num_of_gaussians = num_of_gaussians
        self.proportions_rand = torch.rand(self.num_of_gaussians)
        self.proportions = nn.Parameter(self.proportions_rand/torch.sum(self.proportions_rand))
        
   
    def gaussian_fn(self, M, std, mean):
        #mean 0 - 1
    
        # n = torch.arange(0, M) - (M - 1.0) / 2.0
        n = torch.arange(0, M).to('cuda') - mean
        sig2 = 2 * std * std
        n = n.to('cuda')
        sig2.to('cuda')
        w = torch.exp(-n ** 2 / (sig2 + 1e-7)).to('cuda')
        return w

    def gkern(self, kernlen=256, stdx=0.5, stdy=0.5, meanx= 0.5, meany= 0.5):
        """Returns a 2D Gaussian kernel array."""
        stdx = stdx*kernlen
        stdy = stdy*kernlen
        meanx = meanx*kernlen
        meany = meany*kernlen
        gkern1d_x = self.gaussian_fn(kernlen, std=stdx, mean = meanx) 
        gkern1d_y = self.gaussian_fn(kernlen, std=stdy, mean = meany)
        gkern2d = torch.outer(gkern1d_x, gkern1d_y)
        gkern2d = gkern2d[None, :, :]
        gkern2d = gkern2d.expand(3, kernlen, kernlen)
        return gkern2d
    
    def combine_gaussians(self):
        
        return 
    
    def forward(self, sigmas):
        final_out = []
        if self.num_of_gaussians==1:
            output  = [self.gkern(self.input_size, sigmas[i][0], sigmas[i][1], sigmas[i][2], sigmas[i][3])[None, :, :, :] for i in range(sigmas.shape[0])]
        else:
            output = []
            for i in range(sigmas.shape[0]):             
                output_temp = self.proportions[0]*self.gkern(self.input_size, sigmas[i][0])
                for j in range(1, sigmas.shape[-1]):
                    output_temp += self.proportions[j]*self.gkern(self.input_size, sigmas[i][j])
                
                output+= [output_temp[None, :, :, :]]

        final_out.append(torch.concat(output))
        return final_out

 