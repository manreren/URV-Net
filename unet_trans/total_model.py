import torch
from torch import nn
import torch.nn.functional as F
from unet_SE import Unet
from Blocks import RFB_FRM
from RFB_unet_fusion import RFB_UNET_FUSION
from ViT_Non_local import ImageEncoderViT,Nonlocal
class ALL_net(nn.Module):
    def __init__(self):
        super(ALL_net,self).__init__()
        self.unet_se = Unet(12)
        self.con_rfb = nn.Conv2d(3,16,1,1,0)
        self.rfb = RFB_FRM.BasicRFB(16,32)
        self.rfb_unet_fusion = RFB_UNET_FUSION(128,128)
        self.vit = ImageEncoderViT()
        self.Nonlocal = Nonlocal(128)
        self.conv = nn.Conv2d(256,1,1,1,0)
    def forward(self, x):
        x1 = self.unet_se(x)#[1, 128, 256, 256]
        x2 = self.con_rfb(x)
        x2 = self.rfb(x2)#[1, 128, 512, 512]
        x3 = self.rfb_unet_fusion(x2,x1)#[1, 128, 256, 256]
        x4 = self.vit(x)
        x4 = self.Nonlocal(x4)#[1, 128, 256, 256]
        x5 = torch.cat([x3,x4],dim=1)#[1, 256, 256, 256]
        out = F.interpolate(x5,scale_factor=2)#[1, 256, 512, 512]
        out = self.conv(out)
        return out

if __name__ == "__main__":
    rgb = torch.randn(1,3,512,512)
    net = ALL_net()
    out = net(rgb)
    print(out.shape)