import torch
from torch import nn
import torch.nn.functional as F
from Blocks import RFB_FRM
class ALL_net(nn.Module):
    def __init__(self):
        super(ALL_net,self).__init__()
        self.con_rfb = nn.Conv2d(3,16,1,1,0)
        self.rfb = RFB_FRM.BasicRFB(16,32)
        self.conv = nn.Conv2d(128,1,1,1,0)
    def forward(self, x):
        x2 = self.con_rfb(x)
        x2 = self.rfb(x2)#[1, 128, 512, 512]
        out = self.conv(x2)
        return out

if __name__ == "__main__":
    rgb = torch.randn(1,3,512,512)
    net = ALL_net()
    out = net(rgb)
    print(out.shape)