import torch
from torch import nn
import torch.nn.functional as F
from ViT_Non_local import ImageEncoderViT,Nonlocal
class ALL_net(nn.Module):
    def __init__(self):
        super(ALL_net,self).__init__()
        self.vit = ImageEncoderViT()
        self.Nonlocal = Nonlocal(128)
        self.conv = nn.Conv2d(256,1,1,1,0)
    def forward(self, x):
        x1 = self.vit(x)
        x2 = self.Nonlocal(x1)#[1, 128, 256, 256]
        x3 = torch.cat([x1,x2],dim=1)#[1, 256, 256, 256]
        out = F.interpolate(x3,scale_factor=2)#[1, 256, 512, 512]
        out = self.conv(out)
        return out