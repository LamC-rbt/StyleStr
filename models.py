import torch
from torchvision import models


class ProgressiveTransformerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Non-linearity
        self.relu = torch.nn.ReLU()

        # Down-sampling convolution layers
        num_of_channels = [3, 32, 64, 128]
        kernel_sizes = [9, 3, 3]
        stride_sizes = [1, 2, 2]
        self.conv1 = ConvLayer(num_of_channels[0], num_of_channels[1], kernel_size=kernel_sizes[0], stride=stride_sizes[0])
        self.in1 = torch.nn.InstanceNorm2d(num_of_channels[1], affine=True)
        self.conv2 = ConvLayer(num_of_channels[1], num_of_channels[2], kernel_size=kernel_sizes[1], stride=stride_sizes[1])
        self.in2 = torch.nn.InstanceNorm2d(num_of_channels[2], affine=True)
        self.conv3 = ConvLayer(num_of_channels[2], num_of_channels[3], kernel_size=kernel_sizes[2], stride=stride_sizes[2])
        self.in3 = torch.nn.InstanceNorm2d(num_of_channels[3], affine=True)

        res_block_num_of_filters = 128
        self.res1 = ResidualBlock(res_block_num_of_filters)
        self.res2 = ResidualBlock(res_block_num_of_filters)
        
        self.recurrent_style_block = RecurrentStyleBlock(res_block_num_of_filters)

        # Up-sampling convolution layers
        num_of_channels.reverse()
        kernel_sizes.reverse()
        stride_sizes.reverse()
        self.up1 = UpsampleConvLayer(num_of_channels[0], num_of_channels[1], kernel_size=kernel_sizes[0], stride=stride_sizes[0])
        self.in4 = torch.nn.InstanceNorm2d(num_of_channels[1], affine=True)
        self.up2 = UpsampleConvLayer(num_of_channels[1], num_of_channels[2], kernel_size=kernel_sizes[1], stride=stride_sizes[1])
        self.in5 = torch.nn.InstanceNorm2d(num_of_channels[2], affine=True)
        self.up3 = ConvLayer(num_of_channels[2], num_of_channels[3], kernel_size=kernel_sizes[2], stride=stride_sizes[2])

    def forward(self, x, t=3, return_all=False):
        t = max(1, int(t))
        if return_all:
            t = 5
            out = []
        
        # Encoder
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        
        y = self.res1(y)
        y = self.res2(y)
        base_features = y
        
        batch_size, channels, height, width = base_features.shape
        state = torch.zeros_like(base_features)
        
        current_features = base_features
        for _ in range(t):
            current_features, state = self.recurrent_style_block(base_features, current_features, state)
            if return_all:
                out.append(current_features)
        # if training
        if return_all:
            current_features = torch.cat(out, dim=0) ### [batch*t, c, h, w]
        # Decoder
        y = self.relu(self.in4(self.up1(current_features)))
        y = self.relu(self.in5(self.up2(y)))
        return self.up3(y)


class RecurrentStyleBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.base_conv = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.base_norm = torch.nn.InstanceNorm2d(channels, affine=True)
        
        self.current_conv = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.current_norm = torch.nn.InstanceNorm2d(channels, affine=True)
        
        self.state_conv = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.state_norm = torch.nn.InstanceNorm2d(channels, affine=True)
        
        self.fusion_conv = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.fusion_norm = torch.nn.InstanceNorm2d(channels, affine=True)
        
        self.content_gate = torch.nn.Parameter(torch.ones(1) * 0.8)
        self.style_gate = 1
        
        self.relu = torch.nn.ReLU()
        
    def forward(self, base_features, current_features, state):
        base_out = self.relu(self.base_norm(self.base_conv(base_features)))
        current_out = self.relu(self.current_norm(self.current_conv(current_features)))
        state_out = self.relu(self.state_norm(self.state_conv(state)))
        
        new_state = current_out + state_out
        
        content_importance = torch.sigmoid(self.content_gate)
        style_importance = self.style_gate
        
        combined = content_importance * base_out + style_importance * new_state
        
        output = self.relu(self.fusion_norm(self.fusion_conv(combined)))
        
        return output, new_state


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        return self.conv2d(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        kernel_size = 3
        stride_size = 1
        self.conv1 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=stride_size)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=stride_size)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual  # modification: no ReLu after the addition


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.upsampling_factor = stride
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, stride=1)

    def forward(self, x):
        if self.upsampling_factor > 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsampling_factor, mode='nearest')
        return self.conv2d(x)
