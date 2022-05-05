import torch
from typing import OrderedDict, Tuple, List
from torch import nn
import torchvision.models as models
import torchvision.ops as ops
import numpy as np

class KevinLayer(nn.Sequential):
    def __init__(self,in_channels,out_channels, stride = 1, padding = 1, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
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
        
        self.P1_layer = KevinLayer(64, 64)
        self.P2_layer = KevinLayer(128, 64)
        self.P3_layer = KevinLayer(256, 64)
        self.P4_layer = KevinLayer(512, 64)
        self.P5_layer  = KevinLayer(256, 64)
        self.P6_layer  = KevinLayer(256, 64)


        

        
        
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

        #print('features_dict type ',list(features_dict.values()))

        P6 = features_dict["feat5"]
        P5 = features_dict["feat4"]
        P4 = features_dict["feat3"]
        P3 = features_dict["feat2"]
        P2 = features_dict["feat1"]
        P1 = features_dict["feat0"]
        

        P1 = self.P1_layer(P1)
        print('P1 shape ', P1.shape) # 64
        P2 = self.P2_layer(P2)
        print('P2 shape ', P2.shape) # 128 -> 64
        P3 = self.P3_layer(P3)
        print('P3 shape ', P3.shape) # 256 -> 64
        P4 = self.P4_layer(P4)
        print('P4 shape ', P4.shape) # 512 -> 64
        P5 = self.P5_layer(P5)
        print('P5 shape ', P5.shape) # 256 -> 64
        P6 = self.P6_layer(P6)
        print('P6 shape ', P6.shape) # 256 -> 64


        print('\n')
        
        blue_layer = KevinLayer(256,512).cuda()
        green_layer = KevinLayer(512,256).cuda()
        purple_layer = KevinLayer(256,128).cuda()
        red_layer = KevinLayer(128,64).cuda()
        pink_layer = KevinLayer(64,64).cuda()
        
        yellow_layer = KevinLayer(256,256).cuda() 
        
        for i in range(3):
            # DOWN
            blue_node = (P6 + P5).cuda() # 256 + 256 -> 512
            blue_node = blue_layer(blue_node)

            green_node = (P4 + blue_node) # 512 + 512 - 256
            green_node = green_layer(green_node)

            print('---------GREEEN-----------')
            print('green node shape ', green_node.shape)
            print('P3 shape ', P3.shape) 

            purple_node = (P3 + green_node) # 256 + 256 -> 128
            purple_node = purple_layer(purple_node)


            red_node = (P2 + purple_node) # 128 + 128 -> 64
            red_node =red_layer(red_node)
            
            
            pink_node = (P1 + red_node) # 64 + 64 -> ?
            pink_node = pink_layer(pink_node)

            # UP
            red_node = (red_node + P2 + pink_node) # 128 + 128 + 64
            layer(red_node)
            purple_node = (purple_node + P3 + red_node)
            layer(purple_node)
            green_node = (green_node + P4 + purple_node)
            layer(green_node)
            blue_node = (blue_node + P5 + green_node)
            layer(blue_node)
            yellow_node = (P6 + blue_node)
            layer(yellow_node)
        

            out_features = [yellow_node, blue_node, green_node, purple_node, red_node]

        for idx, feature in enumerate(out_features.values()):
                    out_channel = self.out_channels[idx]
                    h, w = self.output_feature_shape[idx]
                    expected_shape = (out_channel, h, w)
                    assert feature.shape[1:] == expected_shape, \
                        f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
                    assert len(out_features) == len(self.output_feature_shape),\
                        f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

        

