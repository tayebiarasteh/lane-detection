#UNet Implementation
#Refer to the block diagram and build the UNet Model
#You would notice many blocks are repeating with only changes in paramaters like input and output channels
#Make use of the remaining classes to construct these repeating blocks. You may follow the order of ToDo specified
#above each class while writing the code.


#Additional Task: 
#How are wieghts inintialized in this model?
#Read about the various inintializers available in pytorch
#Define a function to inintialize weights in your model
#and create experiments using different initializers.
#Set the name of your experiment accordingly as 
#this initializer information will not be available
#in config file for later reference.
#You can also implement some parts of this task in other
#scripts like Training.py


import torch
import torch.nn as nn
import torch.nn.functional as F

#ToDo 5
class UNet(nn.Module):
    def __init__(self, n_in_channels=3, n_out_classes=2):
        super(UNet, self).__init__()
        #Create object for the components of the network. You may also make use of inconv,up,down and outconv classes defined below.
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
        #Apply input_tensor on object from init function and return the output_tensor 
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
#ToDo 1: Implement double convolution components that you see repeating throughout the architecture.
class double_conv(nn.Module):
    #(conv => Batch Normalization => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        #Create object for the components of the block
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),     # N_features?
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),     # N_features?
            nn.ReLU(inplace=True)
        )



    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
#         input_tensor = input_tensor.type('torch.cuda.FloatTensor')
        output_tensor = self.conv(input_tensor)

        return output_tensor

#ToDo 2: Implement input block
class inconv(nn.Module):
    #Input Block
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        #Create object for the components of the block. You may also make use of double_conv defined above
        self.in_double_conv1 = double_conv(in_ch, out_ch)


    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        output_tensor = self.in_double_conv1(input_tensor)

        
        return output_tensor

#ToDo 2: Implement generic down block
class down(nn.Module):
    #Down Block
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        #Create object for the components of the block.You may also make use of double_conv defined above.
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.down_double_conv1 = double_conv(in_ch, out_ch)


    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        input_tensor = self.maxpool1(input_tensor)
        output_tensor = self.down_double_conv1(input_tensor)


        return output_tensor

#ToDo 3: Implement generic up block
class up(nn.Module):
    #Up Block
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        
        # Create an object for the upsampling operation
        self.upsample1 = nn.Upsample(scale_factor=2)
        
        #Create an object for the remaining components of the block.You may also make use of double_conv defined above.
        self.up_double_conv1 = double_conv(in_ch, out_ch)
        



    def forward(self, input_tensor_1, input_tensor_2):
        #Upsample the input_tensor_1
#         import pdb; pdb.set_trace()
        
        input_tensor_1 = self.upsample1(input_tensor_1)
        
        #Make sure that upsampled tensor and input_tensor_2 have same size for all dimensions that are not concatenated in next step. You may use the method pad() from torch.nn.functional.
#         import pdb; pdb.set_trace()
        #padding = (0, 0, 0, input_tensor_1[0], 0, input_tensor_1[0])
        #input_tensor_1 = F.pad(input_tensor_1, padding, "constant", 0)
        
        #Concatenation of the  upsampled result and input_tensor_2
        input_tensor = torch.cat((input_tensor_1 , input_tensor_2), 1)


        #Apply concatenated result to the object containing remaining components of the block and return result
        output_tensor = self.up_double_conv1(input_tensor)
        


        return output_tensor

#ToDo 4: Implement out block
class outconv(nn.Module):
    #Out Block
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        #Create object for the components of the block
        self.conv_out = nn.Conv2d(in_ch, out_ch, kernel_size=1)


    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        output_tensor = self.conv_out(input_tensor)

        return output_tensor
