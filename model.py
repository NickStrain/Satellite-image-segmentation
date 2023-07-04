import torch
import torch.nn as nn
from torchvision.models import  resnet18
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resmodel = resnet18(pretrained=True).to(device)

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook
def double_conv(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

def up_conv(in_channel,out_channel):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2),
        nn.ReLU(inplace=True)
    )

class unet(nn.Module):
    def __init__(self,out_channel=12):
        super().__init__()
        self.model = resmodel
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.bottleneck = double_conv(512,1024)


        self.up1 = up_conv(1024,512)
        self.merg1 = double_conv(512*2,512)
        self.up2 = up_conv(512,256)
        self.merg2 = double_conv(256*2,256)
        self.up3 = up_conv(256,128)
        self.merg3 = double_conv(128*2,128)
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128,64,3,2,1)
                   ,nn.ReLU(inplace=True))
        self.merg4 = double_conv(64*2,64)
        self.final = nn.Conv2d(64,out_channel,kernel_size=1)

    def forward(self,x):
        a = resmodel.layer1.register_forward_hook(get_activation('layer1'))
        a = resmodel.layer2.register_forward_hook(get_activation('layer2'))
        a = resmodel.layer3.register_forward_hook(get_activation('layer3'))
        a = resmodel.layer4.register_forward_hook(get_activation('layer4'))
        a = resmodel(x)
        x1 = activation['layer1']
        x2 = activation['layer2']
        x3 = activation['layer3']
        x4 = activation['layer4']

        x = self.pool(x4)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.merg1(torch.cat([x,x4],1))
        x = self.up2(x)
        x = self.merg2(torch.cat([x,x3],1))
        x = self.up3(x)
        x = self.merg3(torch.cat([x,x2],1))
        x = self.up4(x)
        x = self.merg4(torch.cat([x,x1],1))

        x = self.final(x)

        return x

model= unet().to(device)
x = torch.zeros(1, 3, 572, 572).to(device)
pre= model(x)
print(pre)







