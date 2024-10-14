import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from scipy import ndimage

from nets.pvtv2 import pvt_v2_b2
from nets.decoders import Decoder
#from models._polyp.cnn_vit_backbone import Transformer, SegmentationHead

logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
    
class PVT_CASCADE(nn.Module):
    def __init__(self,n_channels=3, n_classes=1):
        super(PVT_CASCADE, self).__init__()
        
        self.in_channels = n_channels
        
        
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './nets/pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = Decoder(channels = 128, dims=[512, 320, 128, 64], n_class=n_classes)
        # Prediction heads initialization
        #self.out_head5 = nn.Conv2d(1, n_class, 1)
        #self.out_head1 = nn.Conv2d(1, n_class, 1)
        #self.out_head2 = nn.Conv2d(1, n_class, 1)
        #self.out_head3 = nn.Conv2d(1, n_class, 1)
        #self.out_head4 = nn.Conv2d(1, n_class, 1)
        
      

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        x = self.conv(x)
        #if x.size()[1] == 1:
        #    x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)

        shape = x.size()[2:]
        #print('Shape of x is ',x.shape)
        
        
        # decoder
        P5, P4, P3, P2, P1 = self.decoder(x4, x3, x2, x1, shape)
        # prediction heads  
        #P1 = self.out_head1(P1)
        #P2 = self.out_head2(P2)
        #P3 = self.out_head3(P3)
        #P4 = self.out_head4(P4)
        #P5 = self.out_head5(P5)
        
            
        return  P5,P4,P3,P2,P1

        
if __name__ == '__main__':
    model = PVT_CASCADE().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    p1, p2, p3, p4 = model(input_tensor)
    print(p1.size(), p2.size(), p3.size(), p4.size())

