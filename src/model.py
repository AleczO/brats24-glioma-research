from monai.networks.nets import UNet
import torch

def get_model():
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=2
    )
    return model