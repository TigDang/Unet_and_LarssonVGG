# Unet_and_LarssonVGG
## About
This is my thesis project where we try to use Unet instead VGG-16 in colorization method Larrson et al. (https://arxiv.org/pdf/1603.06668.pdf)

## Install
- clone that repo
- install requirements from file
- run test.py or train.py with options

## VGG

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
        
            Conv2d-1           [-1, 64, 64, 64]             640
       BatchNorm2d-2           [-1, 64, 64, 64]             128
         LeakyReLU-3           [-1, 64, 64, 64]               0
          VGGLayer-4           [-1, 64, 64, 64]               0
            Conv2d-5           [-1, 64, 64, 64]          36,928
       BatchNorm2d-6           [-1, 64, 64, 64]             128
         LeakyReLU-7           [-1, 64, 64, 64]               0
          VGGLayer-8           [-1, 64, 64, 64]               0
         MaxPool2d-9           [-1, 64, 32, 32]               0
           Conv2d-10          [-1, 128, 32, 32]          73,856
      BatchNorm2d-11          [-1, 128, 32, 32]             256
        LeakyReLU-12          [-1, 128, 32, 32]               0
         VGGLayer-13          [-1, 128, 32, 32]               0
           Conv2d-14          [-1, 128, 32, 32]         147,584
      BatchNorm2d-15          [-1, 128, 32, 32]             256
        LeakyReLU-16          [-1, 128, 32, 32]               0
         VGGLayer-17          [-1, 128, 32, 32]               0
        MaxPool2d-18          [-1, 128, 16, 16]               0
           Conv2d-19          [-1, 256, 16, 16]         295,168
      BatchNorm2d-20          [-1, 256, 16, 16]             512
        LeakyReLU-21          [-1, 256, 16, 16]               0
         VGGLayer-22          [-1, 256, 16, 16]               0
           Conv2d-23          [-1, 256, 16, 16]         590,080
      BatchNorm2d-24          [-1, 256, 16, 16]             512
        LeakyReLU-25          [-1, 256, 16, 16]               0
         VGGLayer-26          [-1, 256, 16, 16]               0
           Conv2d-27          [-1, 256, 16, 16]         590,080
      BatchNorm2d-28          [-1, 256, 16, 16]             512
        LeakyReLU-29          [-1, 256, 16, 16]               0
         VGGLayer-30          [-1, 256, 16, 16]               0
        MaxPool2d-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 512, 8, 8]       1,180,160
      BatchNorm2d-33            [-1, 512, 8, 8]           1,024
        LeakyReLU-34            [-1, 512, 8, 8]               0
         VGGLayer-35            [-1, 512, 8, 8]               0
           Conv2d-36            [-1, 512, 8, 8]       2,359,808
      BatchNorm2d-37            [-1, 512, 8, 8]           1,024
        LeakyReLU-38            [-1, 512, 8, 8]               0
         VGGLayer-39            [-1, 512, 8, 8]               0
           Conv2d-40            [-1, 512, 8, 8]       2,359,808
      BatchNorm2d-41            [-1, 512, 8, 8]           1,024
        LeakyReLU-42            [-1, 512, 8, 8]               0
         VGGLayer-43            [-1, 512, 8, 8]               0
        MaxPool2d-44            [-1, 512, 4, 4]               0
           Conv2d-45            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-46            [-1, 512, 4, 4]           1,024
        LeakyReLU-47            [-1, 512, 4, 4]               0
         VGGLayer-48            [-1, 512, 4, 4]               0
           Conv2d-49            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-50            [-1, 512, 4, 4]           1,024
        LeakyReLU-51            [-1, 512, 4, 4]               0
         VGGLayer-52            [-1, 512, 4, 4]               0
           Conv2d-53            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-54            [-1, 512, 4, 4]           1,024
        LeakyReLU-55            [-1, 512, 4, 4]               0
         VGGLayer-56            [-1, 512, 4, 4]               0
        MaxPool2d-57            [-1, 512, 2, 2]               0
           Conv2d-58           [-1, 1024, 2, 2]       4,719,616
      BatchNorm2d-59           [-1, 1024, 2, 2]           2,048
        LeakyReLU-60           [-1, 1024, 2, 2]               0
         VGGLayer-61           [-1, 1024, 2, 2]               0
           Conv2d-62           [-1, 1024, 2, 2]       9,438,208
      BatchNorm2d-63           [-1, 1024, 2, 2]           2,048
        LeakyReLU-64           [-1, 1024, 2, 2]               0
         VGGLayer-65           [-1, 1024, 2, 2]               0
         Upsample-66           [-1, 64, 64, 64]               0
         Upsample-67          [-1, 128, 64, 64]               0
         Upsample-68          [-1, 256, 64, 64]               0
         Upsample-69          [-1, 512, 64, 64]               0
         Upsample-70          [-1, 512, 64, 64]               0
         Upsample-71         [-1, 1024, 64, 64]               0
         Upsample-72         [-1, 1024, 64, 64]               0
           Conv2d-73         [-1, 1024, 64, 64]       3,605,504
      BatchNorm2d-74         [-1, 1024, 64, 64]           2,048
      
Total params: 32,491,456
Trainable params: 32,491,456
Non-trainable params: 0

Input size (MB): 0.02
Forward/backward pass size (MB): 208.95
Params size (MB): 123.95
Estimated Total Size (MB): 332.91


## Unet

----------------------------------------------------------------
        Layer (type)               Output Shape         Param # 
        
            Conv2d-1           [-1, 64, 64, 64]             576
       BatchNorm2d-2           [-1, 64, 64, 64]             128
         LeakyReLU-3           [-1, 64, 64, 64]               0
            Conv2d-4           [-1, 64, 64, 64]          36,864
       BatchNorm2d-5           [-1, 64, 64, 64]             128
         LeakyReLU-6           [-1, 64, 64, 64]               0
        DoubleConv-7           [-1, 64, 64, 64]               0
         MaxPool2d-8           [-1, 64, 32, 32]               0
            Conv2d-9          [-1, 128, 32, 32]          73,728
      BatchNorm2d-10          [-1, 128, 32, 32]             256
        LeakyReLU-11          [-1, 128, 32, 32]               0
           Conv2d-12          [-1, 128, 32, 32]         147,456
      BatchNorm2d-13          [-1, 128, 32, 32]             256
        LeakyReLU-14          [-1, 128, 32, 32]               0
       DoubleConv-15          [-1, 128, 32, 32]               0
             Down-16          [-1, 128, 32, 32]               0
        MaxPool2d-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 256, 16, 16]         294,912
      BatchNorm2d-19          [-1, 256, 16, 16]             512
        LeakyReLU-20          [-1, 256, 16, 16]               0
           Conv2d-21          [-1, 256, 16, 16]         589,824
      BatchNorm2d-22          [-1, 256, 16, 16]             512
        LeakyReLU-23          [-1, 256, 16, 16]               0
       DoubleConv-24          [-1, 256, 16, 16]               0
             Down-25          [-1, 256, 16, 16]               0
        MaxPool2d-26            [-1, 256, 8, 8]               0
           Conv2d-27            [-1, 512, 8, 8]       1,179,648
      BatchNorm2d-28            [-1, 512, 8, 8]           1,024
        LeakyReLU-29            [-1, 512, 8, 8]               0
           Conv2d-30            [-1, 512, 8, 8]       2,359,296
      BatchNorm2d-31            [-1, 512, 8, 8]           1,024
        LeakyReLU-32            [-1, 512, 8, 8]               0
       DoubleConv-33            [-1, 512, 8, 8]               0
             Down-34            [-1, 512, 8, 8]               0
        MaxPool2d-35            [-1, 512, 4, 4]               0
           Conv2d-36           [-1, 1024, 4, 4]       4,718,592
      BatchNorm2d-37           [-1, 1024, 4, 4]           2,048
        LeakyReLU-38           [-1, 1024, 4, 4]               0
           Conv2d-39           [-1, 1024, 4, 4]       9,437,184
      BatchNorm2d-40           [-1, 1024, 4, 4]           2,048
        LeakyReLU-41           [-1, 1024, 4, 4]               0
       DoubleConv-42           [-1, 1024, 4, 4]               0
             Down-43           [-1, 1024, 4, 4]               0
  ConvTranspose2d-44            [-1, 512, 8, 8]       2,097,664
           Conv2d-45            [-1, 512, 8, 8]       4,718,592
      BatchNorm2d-46            [-1, 512, 8, 8]           1,024
        LeakyReLU-47            [-1, 512, 8, 8]               0
           Conv2d-48            [-1, 512, 8, 8]       2,359,296
      BatchNorm2d-49            [-1, 512, 8, 8]           1,024
        LeakyReLU-50            [-1, 512, 8, 8]               0
       DoubleConv-51            [-1, 512, 8, 8]               0
               Up-52            [-1, 512, 8, 8]               0
  ConvTranspose2d-53          [-1, 256, 16, 16]         524,544
           Conv2d-54          [-1, 256, 16, 16]       1,179,648
      BatchNorm2d-55          [-1, 256, 16, 16]             512
        LeakyReLU-56          [-1, 256, 16, 16]               0
           Conv2d-57          [-1, 256, 16, 16]         589,824
      BatchNorm2d-58          [-1, 256, 16, 16]             512
        LeakyReLU-59          [-1, 256, 16, 16]               0
       DoubleConv-60          [-1, 256, 16, 16]               0
               Up-61          [-1, 256, 16, 16]               0
  ConvTranspose2d-62          [-1, 128, 32, 32]         131,200
           Conv2d-63          [-1, 128, 32, 32]         294,912
      BatchNorm2d-64          [-1, 128, 32, 32]             256
        LeakyReLU-65          [-1, 128, 32, 32]               0
           Conv2d-66          [-1, 128, 32, 32]         147,456
      BatchNorm2d-67          [-1, 128, 32, 32]             256
        LeakyReLU-68          [-1, 128, 32, 32]               0
       DoubleConv-69          [-1, 128, 32, 32]               0
               Up-70          [-1, 128, 32, 32]               0
  ConvTranspose2d-71           [-1, 64, 64, 64]          32,832
           Conv2d-72           [-1, 64, 64, 64]          73,728
      BatchNorm2d-73           [-1, 64, 64, 64]             128
        LeakyReLU-74           [-1, 64, 64, 64]               0
           Conv2d-75           [-1, 64, 64, 64]          36,864
      BatchNorm2d-76           [-1, 64, 64, 64]             128
        LeakyReLU-77           [-1, 64, 64, 64]               0
       DoubleConv-78           [-1, 64, 64, 64]               0
               Up-79           [-1, 64, 64, 64]               0
           Conv2d-80         [-1, 1024, 64, 64]          66,560
          OutConv-81         [-1, 1024, 64, 64]               0
          
Total params: 31,102,976
Trainable params: 31,102,976
Non-trainable params: 0

Input size (MB): 0.02
Forward/backward pass size (MB): 127.69
Params size (MB): 118.65
Estimated Total Size (MB): 246.35
