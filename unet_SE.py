import torch
import torch.nn as nn
from Blocks import SEBlock
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class up(nn.Module):
    def __init__(self, in_ch, out_ch, Transpose=False):
        super(up, self).__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        else:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch//2, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True))
        self.conv = DoubleConv(in_ch, out_ch)
        self.up.apply(self.init_weights)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))
        x = torch.cat([x2,x1], dim=1)
        x = self.conv(x)
        return x
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias,0)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class Unet(nn.Module):
    def __init__(self, out_ch):
        super(Unet, self).__init__()
        self.display_names = ['loss_stack', 'matrix_iou_stack']
        self.bce_loss = nn.BCELoss()
        self.down1 = Down(3, 128)
        # print(list(self.down1.parameters()))
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.drop3 = nn.Dropout2d(0.5)
        self.down4 = Down(512, 1024)
        self.drop4 = nn.Dropout2d(0.5)
        self.up1 = up(1024, 512, False)
        self.up2 = up(512, 256, False)
        self.up3 = up(256, 128, False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.SEBlock4 = SEBlock.SE_Block(128,128)
        self.SEBlock3 = SEBlock.SE_Block(256,56)
        self.SEBlock2 = SEBlock.SE_Block(512,512)
        self.SEBlock1 = SEBlock.SE_Block(1024,1024)
    def forward(self,x):
        x1=x
        x2 = self.down1(x1)#[1, 128, 256, 256]
        x2 = self.SEBlock4(x2)
        x3 = self.down2(x2)#[1, 256, 128, 128]
        x3 = self.SEBlock3(x3)
        x4 = self.down3(x3)#[1, 512, 64, 64]
        x4 = self.SEBlock2(x4)
        x4 = self.drop3(x4)
        x5 = self.down4(x4)#[1, 1024, 32, 32]
        x5 = self.SEBlock1(x5)
        x5 = self.drop4(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        return x

if __name__ == "__main__":
    import torch as t
    rgb = t.randn(1, 3, 512, 512)
    net = Unet(12)

    out = net(rgb)

    print(out.shape)