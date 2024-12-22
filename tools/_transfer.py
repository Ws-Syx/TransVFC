import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, activate="relu"):
        super(BottleneckBlock, self).__init__()
        
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True) if activate=="relu" else nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(mid_ch)
        self.relu2 = nn.ReLU(inplace=True) if activate=="relu" else nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = nn.BatchNorm2d(out_ch)
        self.relu3 = nn.ReLU(inplace=True) if activate=="relu" else nn.Sigmoid()
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu3(out+residual)
        
        return out

class DownNet(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(DownNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        
        self.down_sample = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.res2 = nn.Sequential(
            BottleneckBlock(in_ch=mid_channel, mid_ch=mid_channel, out_ch=out_channel),
            BottleneckBlock(in_ch=out_channel, mid_ch=mid_channel, out_ch=out_channel),
            BottleneckBlock(in_ch=out_channel, mid_ch=mid_channel, out_ch=out_channel),
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.down_sample(x)
        x = self.res2(x)
        return x

class UpNet(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, activate="relu"):
        super(UpNet, self).__init__()
        
        self.res1 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            BottleneckBlock(in_ch=mid_channel, mid_ch=mid_channel, out_ch=mid_channel, activate=activate),
            BottleneckBlock(in_ch=mid_channel, mid_ch=mid_channel, out_ch=mid_channel, activate=activate),
        )
        
        self.res2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            BottleneckBlock(in_ch=mid_channel, mid_ch=mid_channel, out_ch=out_channel, activate=activate),
            BottleneckBlock(in_ch=out_channel, mid_ch=mid_channel, out_ch=out_channel, activate=activate),
        )
    
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return x

class FeatureSpaceTransfer(torch.nn.Module):
    def __init__(self, in_channel=256, out_channel=256, mid_channel=64):
        super(FeatureSpaceTransfer, self).__init__()
        
        self.branch1_up = UpNet(in_channel=in_channel, out_channel=3, mid_channel=mid_channel, activate="sigmoid")
        self.branch1_down = DownNet(in_channel=3, out_channel=out_channel, mid_channel=mid_channel)
        
        self.branch2_down = DownNet(in_channel=in_channel, out_channel=mid_channel, mid_channel=mid_channel)
        self.branch2_up = UpNet(in_channel=mid_channel, out_channel=out_channel, mid_channel=mid_channel, activate="relu")
        
        self.branch3 = nn.Sequential(
            BottleneckBlock(in_ch=in_channel, mid_ch=mid_channel, out_ch=out_channel, activate="relu"),
            BottleneckBlock(in_ch=out_channel, mid_ch=mid_channel, out_ch=out_channel, activate="relu"),
            BottleneckBlock(in_ch=out_channel, mid_ch=mid_channel, out_ch=out_channel, activate="relu")
        )
        
        
        self.fuse = nn.Conv2d(out_channel*3, out_channel, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        recon_image = self.branch1_up(x)
        y1 = self.branch1_down(recon_image)
        
        y2 = self.branch2_down(x)
        y2 = self.branch2_up(y2)
        
        y3 = self.branch3(x)
        
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.fuse(y)
        return y, recon_image
        
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# test 
if __name__ == "__main__":
    # create the model
    model = FeatureSpaceTransfer().cuda()
    print(f"Total trainable parameters in the model: {count_parameters(model)}")
    
    print(f"Total trainable parameters in the model.up: {count_parameters(model.branch1_up)}")
    print(f"Total trainable parameters in the model.down: {count_parameters(model.branch1_down)}")
    
    print(f"Total trainable parameters in the model.up: {count_parameters(model.branch2_up)}")
    print(f"Total trainable parameters in the model.down: {count_parameters(model.branch2_down)}")
    
    print(f"Total trainable parameters in the branch3: {count_parameters(model.branch3)}")

    # print(model)
    # create a input data
    input_data = torch.randn((4, 256, 64, 64)).cuda()
    output_data, output_image = model(input_data)
    print("input_data.shape: ", input_data.shape)
    print("output_data.shape: ", output_data.shape)
    print("output_image.shape: ", output_image.shape)