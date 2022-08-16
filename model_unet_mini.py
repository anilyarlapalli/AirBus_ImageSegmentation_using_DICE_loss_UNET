
import torch
import torchvision
import torch.nn as nn

def ImageSegmentation(num_classes):
    PRE_TRAINED_NET = torchvision.models.resnet18(pretrained=True)
    model = nn.Sequential(*list(PRE_TRAINED_NET.children())[:-2])
    model.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    model.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))

    W = bilinear_kernel(num_classes, num_classes, 64)
    model.transpose_conv.weight.data.copy_(W)

    return model

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),torch.arange(kernel_size).reshape(1, -1))
    
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
    kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight