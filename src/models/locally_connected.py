import torch 
import torch.nn as nn 

class LocallyConnected2d(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 output_size, 
                 kernel_size=3,
                 stride=1,
                 padding=1, 
                 bias=True):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
            
        self.out_h, self.out_w = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        k = kernel_size
        in_dim = in_channels * k * k
        positions = self.out_h * self.out_w
        
        self.weight = nn.Parameter(torch.empty(out_channels, positions, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, positions))
        else:
            self.bias = None
            
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        self.unfold = nn.Unfold(kernel_size=k, stride=stride, padding=padding)
        
    def forward(self, x):
        B = x.size(0)
        
        patches = self.unfold(x)
        patches = patches.transpose(1, 2) # (B, positions, in_dim)
        
        out = torch.einsum("bpi,opi->bop", patches, self.weight)
        
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
            
        out = out.view(B, self.out_channels, self.out_h, self.out_w)
        
        return out