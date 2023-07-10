from torch import nn
from . import my_se_block as se 

class MySELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8, se_block_type:str = 'SSE'):
        super(MySELayer, self).__init__()
        
        self.se_block_type = se_block_type
            
        if self.se_block_type == 'CSE':
            self.layer = se.ChannelSpatialSELayer(num_channels=num_channels, reduction_ratio=reduction_ratio)
        elif self.se_block_type == 'SSE':
            self.layer = se.SpatialSELayer(num_channels=num_channels)
        elif self.se_block_type == 'CSSE':
            self.layer = se.ChannelSpatialSELayer(num_channels=num_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        x = self.layer(x)
        return x
