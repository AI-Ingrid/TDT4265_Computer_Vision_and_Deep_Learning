import torch
from typing import Tuple, List
from torch import nn

# TODO: Mulig vi må ta inn nn.Sequential
class FirstLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding = 1):
        super().__init__(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=kernel_size, stride=2, padding=padding),
        nn.ReLU(),
        )

class Layer(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride_one=1, stride_two=2, padding_one=1, padding_two=1):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels= in_channels, kernel_size=kernel_size, stride=stride_one, padding=padding_one),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=kernel_size, stride=stride_two, padding=padding_two),
            nn.ReLU(),
        )
class BasicModel(torch.nn.Module):
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
        
        # Creating CNN
        self.features = nn.ModuleList() # TODO: Should we use ModuleList here or Sequential
        # 1
        self.first_layer = FirstLayer(image_channels, output_channels[0])
        self.features.append(self.first_layer)
        # 2
        self.second_layer = Layer(output_channels[0], 128, output_channels[1])
        self.features.append(self.second_layer)
        # 3
        self.third_layer = Layer(output_channels[1], 265, output_channels[2])
        self.features.append(self.third_layer)
        # 4
        self.fourth_layer = Layer(output_channels[2], 128, output_channels[3])
        self.features.append(self.fourth_layer)
        # 5
        self.fifth_layer = Layer(output_channels[3], 128, output_channels[4])
        self.features.append(self.fifth_layer)
        # 6
        self.last_layer = Layer(output_channels[4], 128, output_channels[5], stride_two=1, padding_two=0)
        self.features.append(self.last_layer)

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
        for layer in self.features:
            x = layer(x)
            out_features.append(x)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

