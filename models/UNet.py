import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_in_channels=3, n_out_classes=2):
        super(UNet, self).__init__()
        self.input_block = inconv(n_in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.output_block = outconv(64, n_out_classes)
        

    def forward(self, input_tensor):
        first_output = self.input_block(input_tensor)
        
        down1_output = self.down1(first_output)
        down2_output = self.down2(down1_output)
        down3_output = self.down3(down2_output)
        down4_output = self.down4(down3_output)
        
        up1_output = self.up1(down4_output, down3_output)
        up2_output = self.up2(up1_output, down2_output)
        up3_output = self.up3(up2_output, down1_output)
        up4_output = self.up4(up3_output, first_output)
        
        output_tensor = self.output_block(up4_output)
        return output_tensor
    
    
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_tensor):
        output_tensor = self.conv(input_tensor)
        return output_tensor


#Input Block
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.in_double_conv1 = double_conv(in_ch, out_ch)

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        output_tensor = self.in_double_conv1(input_tensor)
        return output_tensor

#Down Block
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.down_double_conv1 = double_conv(in_ch, out_ch)

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor
        input_tensor = self.maxpool1(input_tensor)
        output_tensor = self.down_double_conv1(input_tensor)
        return output_tensor

#Up Block
class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up_double_conv1 = double_conv(in_ch, out_ch)
        
    def forward(self, input_tensor_1, input_tensor_2):
        input_tensor_1 = self.upsample1(input_tensor_1)
        #Concatenation of the  upsampled result and input_tensor_2
        input_tensor = torch.cat((input_tensor_1 , input_tensor_2), 1)
        output_tensor = self.up_double_conv1(input_tensor)
        return output_tensor

#Out Block
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv_out = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, input_tensor):
        output_tensor = self.conv_out(input_tensor)
        return output_tensor
