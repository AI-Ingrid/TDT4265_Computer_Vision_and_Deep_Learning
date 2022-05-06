import torch
from typing import OrderedDict, Tuple, List
from torch import nn
import torchvision.models as models
import torchvision.ops as ops
import numpy as np
import torch.nn.functional as F

class KevinLayer(nn.Sequential):
    def __init__(self,in_channels,out_channels, stride = 1, padding = 1, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        )

class ConvLayerWithRelu(nn.Sequential):
    def __init__(self,in_channels,out_channels, stride = 1, padding = 1, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), 
        )



class Layer(nn.Sequential):
    def __init__(self,in_channels,out_channels):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels= in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels= in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels= in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels= in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=3, stride=1, padding=1),
        )


class ResNetBiFPN(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        # Get pretrained Retina Network
        self.model = models.resnet34(pretrained=True)

        # Create two more layers
        self.layer5 = Layer(512, 256)
        self.layer6 = Layer(256, 256)
        
        # Used to reshape the inputs
        self.P1_layer = KevinLayer(256, 64)
        self.P2_layer = KevinLayer(256, 64)
        self.P3_layer = KevinLayer(512, 64)
        self.P4_layer = KevinLayer(256, 64, stride = 2)
        self.P5_layer  = KevinLayer(128, 64, stride = 4)
        self.P6_layer  = KevinLayer(64, 64, stride = 8)

        # Brukes i formel
        self.epsilon = 0.0001

        # td
        self.P1_td_layer = ConvLayerWithRelu(64,64)
        self.P2_td_layer = ConvLayerWithRelu(64,64)
        self.P3_td_layer = ConvLayerWithRelu(64,64)
        self.P4_td_layer = ConvLayerWithRelu(64,64)
        self.P5_td_layer = ConvLayerWithRelu(64,64)

        # out
        self.P2_out_layer = ConvLayerWithRelu(64,64)
        self.P3_out_layer = ConvLayerWithRelu(64,64)
        self.P4_out_layer = ConvLayerWithRelu(64,64)
        self.P5_out_layer = ConvLayerWithRelu(64,64)
        self.P6_out_layer = ConvLayerWithRelu(64,64)

        # Weights
        # Opp og ned
        self.w1 = nn.Parameter(torch.Tensor(2, 5))
        self.w1_relu = nn.ReLU()
        # Til høyre
        self.w2 = nn.Parameter(torch.Tensor(3, 5))
        self.w2_relu = nn.ReLU()
        

        
        
    def forward_first_layer(self, model, image):
        """Executing forward pass for the zeroth Retina Net layer"""
        x = model.conv1(image)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        return x
        


    def forward(self, x):
        #print("out channels ", self.out_channels)
        #print("out feature ", self.output_feature_shape)
        """
        Performing forward pass for a layer at the time and saving every output in an array. 
        The forward functiom should output features with shape:
            [shape(-1, 256, 32, 256),
            shape(-1, 512, 16, 128),
            shape(-1, 1024, 8, 64),
            shape(-1, 2048, 4, 32),
            shape(-1, 2048, 2, 16),
            shape(-1, 2048, 1, 8)]
        When done, the array of outputs is passed into the FPN and the outputs from FPN is returned
        """
        features_dict = OrderedDict()
        out_features = []

        # Layer 0
        x = self.forward_first_layer(self.model , x)

        # Layer 1
        features_dict['feat0'] = self.model.layer1(x)
        
        # Layer 2
        features_dict['feat1'] = self.model.layer2(features_dict['feat0'])

        # Layer 3
        features_dict['feat2'] = self.model.layer3(features_dict['feat1'])

        # Layer 4
        features_dict['feat3'] = self.model.layer4(features_dict['feat2'])

        # Layer 5
        features_dict['feat4'] = self.layer5(features_dict['feat3'])
        
        # Layer 6
        features_dict['feat5'] = self.layer6(features_dict['feat4'])

        P1 = features_dict["feat5"]
        P2 = features_dict["feat4"]
        P3 = features_dict["feat3"]
        P4 = features_dict["feat2"]
        P5 = features_dict["feat1"]
        P6 = features_dict["feat0"]

        # Resizing each Px to have 64 channels
        P1 = self.P1_layer(P1)
        # print('P1 shape ', P1.shape) # 64
        P2 = self.P2_layer(P2)
        # print('P2 shape ', P2.shape) # 128 -> 64
        P3 = self.P3_layer(P3)
        # print('P3 shape ', P3.shape) # 256 -> 64
        P4 = self.P4_layer(P4)
        # print('P4 shape ', P4.shape) # 512 -> 64
        P5 = self.P5_layer(P5)
        # print('P5 shape ', P5.shape) # 256 -> 64
        P6 = self.P6_layer(P6)
        # print('P6 shape ', P6.shape) # 256 -> 64

        # print("P1 shape:", P1.shape)
        # print("P2 shape:", P2.shape)
        # print("P3 shape:", P3.shape)
        # print("P4 shape:", P4.shape)
        # print("P5 shape:", P5.shape)
        # print("P6 shape:", P6.shape)
        
       

        # Top down
        w1 = self.w1_relu(self.w1)
        w1 /=  torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon

        up = nn.Upsample(scale_factor=2, mode='nearest')
        down = nn.MaxPool2d(kernel_size=2)

        #P5_td = self.P5_td_layer((self.w2[0,1] * P5 + self.w1[0,0] * P6)/(self.w1[0,0] + self.w2[0,1] + self.epsilon))
        P6_td = up(P6)
        P5_td = self.P5_td_layer(self.w1[1,0] * P6_td + self.w1[0,0] * up(P5))
        P4_td = self.P4_td_layer(self.w1[1,1] * P5_td + self.w1[0,1] * up(P4))
        P3_td = self.P3_td_layer(self.w1[1,2] * P4_td + self.w1[0,2] * up(P3))
        P2_td = self.P2_td_layer(self.w1[1,3] * P3_td + self.w1[0,3] * up(P2))
        P1_td = self.P1_td_layer(self.w1[1,4] * P2_td + self.w1[0,4] * up(P1))
        
        # print("P1 shape:", P1_td.shape)
        # print("P3 shape:", P3_td.shape)
        # print("P4 shape:", P4_td.shape)
        # print("P5 shape:", P5_td.shape)
        # print("P6 shape:", P6_td.shape)

        # Bottom up
        P1_out = (P1_td)
        P2_out = self.P2_out_layer(self.w2[0,0] * up(P2) + self.w2[1,0] * P2_td + self.w2[2,0] * (P1_out))
        P3_out = self.P3_out_layer(self.w2[0,1] * up(P3) + self.w2[1,1] * P3_td + self.w2[2,1] * (P2_out))
        P4_out = self.P4_out_layer(self.w2[0,2] * up(P4) + self.w2[1,2] * P4_td + self.w2[2,2] * (P3_out))
        P5_out = self.P5_out_layer(self.w2[0,3] * up(P5) + self.w2[1,3] * P5_td + self.w2[2,3] * (P4_out))
        P6_out = self.P6_out_layer(self.w2[0,4] * up(P6) + self.w2[1,4] * P6_td + self.w2[2,4] * P5_out)
        
        up4 = nn.Upsample(scale_factor=4, mode='nearest')
        up2 = nn.Upsample(scale_factor=2, mode='nearest')
        down2 = nn.MaxPool2d(kernel_size=2)
        down4 = nn.MaxPool2d(kernel_size=4)
        down8 = nn.MaxPool2d(kernel_size=8)
        out_features = [up4(P6_out), up2(P5_out), (P4_out), down2(P3_out), down4(P2_out), down8(P1_td)]
        
        # Box prediction net

        """
        # out
        self.P2_out = ConvLayerWithRelu(64,64)
        self.P3_out = ConvLayerWithRelu(64,64)
        self.P4_out = ConvLayerWithRelu(64,64)
        self.P5_out = ConvLayerWithRelu(64,64)
        self.P6_out = ConvLayerWithRelu(64,64)
        
        """

        for idx, feature in enumerate(out_features):
                    out_channel = self.out_channels[idx]
                    h, w = self.output_feature_shape[idx]
                    expected_shape = (out_channel, h, w)
                    assert feature.shape[1:] == expected_shape, \
                        f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
                    assert len(out_features) == len(self.output_feature_shape),\
                        f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

        

