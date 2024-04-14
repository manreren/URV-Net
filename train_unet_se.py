import torch
from torch import nn
from unet_SE import Unet

class ALL_net(nn.Module):
    def __init__(self):
        super(ALL_net,self).__init__()
        self.unet_se = Unet(12)
        self.deconv =nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv = nn.Conv2d(256,1,1,1,0)
    def forward(self, x):
        x1 = self.unet_se(x)#[1, 128, 256, 256]
        x2 = self.deconv(x1)#[1, 256, 512, 512]
        out = self.conv(x2)#[1, 1, 512, 512]
        return out

if __name__ == "__main__":
    rgb = torch.randn(1,3,512,512)
    net = ALL_net()
    out = net(rgb)
    print(out.shape)