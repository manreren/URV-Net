import torch
from torch import nn
class FRM(nn.Module):
    def __init__(self):
        super(FRM).__init__()
        self.conv1 = nn.Conv2d(32,64,3,1,0)
        self.conv2 = nn.Conv2d(64,64,3,1,0)
        self.conv3 = nn.Conv2d(64,128,3,1,0)
        self.conv4 = nn.Conv2d(128,128,3,1,0)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.ReLU()
    def forwar(self,x):
        x1 = self.conv1(x)
        x2 = self.act1(x1)
        x2 = self.conv2(x2)
        x_shotcut1 = x1+x2

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        '''
        in_planes初始化不能是3
        '''
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FRM(nn.Module):
    def __init__(self, in_ch=32, ou_ch=32):
        super(FRM, self).__init__()
        self.in_ch = in_ch
        self.ou_ch =ou_ch
        self.conv1 = nn.Conv2d(in_channels=in_ch,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.ReLU()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.act1(x1)
        x2 = self.conv2(x2)
        x2 = x2 + x1
        x3 = self.conv3(x2)
        x4 = self.act2(x3)
        x4 = self.conv4(x4)
        x4 = x4+x3
        return x4
class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                      relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.FRM = FRM()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        out = self.FRM(out)

        return out#[1, 128, 512, 512]


if __name__=="__main__":
    #需要x的channel是16，输入的时候过一个1*1卷积
    x=torch.randn(1,16,512,512)
    net = BasicRFB(16,32)
    out = net(x)
    print(out.shape)