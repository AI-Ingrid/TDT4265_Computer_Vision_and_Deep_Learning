import torch
from typing import Tuple, List
from torch import nn
import torchvision.models as models

class FirstLayer(nn.Sequential):
    def __init__(self, model):
        super().__init__(
        model.conv1(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        model.bn1(64, eps=1e-05, momentum=0.1, affine=True, track_running_status=True),
        model.relu(inplace=True),
        model.maxpool(kernal_size=3, stride=2,padding=1, dilation=1, ceil_mode=False)
        )

class LastLayer(nn.Sequential):
    def __init__(self, model):
        super().__init__(
            model.relu(),
            model.conv1(128, 128 , kernel_size=3, stride=1, padding=1),
            model.relu(),
            model.conv1(128, 64, kernel_size=4, stride=2, padding=1),
            model.relu(),
        )

class ResNet(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        self.model = torch.torchvision.resnet34(pratrained=True)
        self.resnet = nn.Sequential(
            FirstLayer(self.model),
            *list(torch.torchvision.resnet34(pretrained=True).children())[4:-2],
            LastLayer(self.model),
            )
        print(self.resnet.model())
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Create a FPN with all the outputs
        self.feature_pyramid_net = torch.torchvision.ops.FeaturesPyramidNetwork()

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        # Peform forward on each layer in the network
        for layer in self.resnet: # check what "layer" actually is, might be iterating throug something else
            x = layer(x)
            # Save all by adding them to a list 
            out_features.append(x)
        
        # Add the output of all layers and put it into the FPN
        out_features = self.feature_pyramid_net(out_features)
    
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        
        # Return a list/typle of all output features from the FPN 
        return tuple(out_features)

