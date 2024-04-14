import torch
from torch import nn
from models import ASPP

class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=in_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.bn1(out)
        out = self.point_conv(out)
        out = self.bn2(out)
        out = self.act(out)
        return out


class RFB_UNET_FUSION(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(RFB_UNET_FUSION, self).__init__()
        self.in_ch = in_ch
        self.ou_ch = ou_ch
        self.conv = DEPTHWISECONV(in_ch, ou_ch)
        self.aspp = ASPP(128,128,1)

    def forward(self, x1, x2):  # x1[128,256,256]
        x1 = self.conv(x1)  # [128,512,512]
        out = x1 + x2
        out = self.aspp(out)
        return out


if __name__=="__main__":
    x1 = torch.randn(1, 128, 256, 256)
    x2 = torch.randn(1,128,512,512)
    net = RFB_UNET_FUSION(128,128)
    out = net(x2,x1)
    print(out.shape)