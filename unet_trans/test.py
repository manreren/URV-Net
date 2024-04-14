import torch
import torch.nn as nn

deconv = nn.Sequential(
nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
)

if __name__=="__main__":
    x=torch.randn(1,256,32,32)
    out = deconv(x)
    print(out.shape)