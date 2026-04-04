from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    NormalizeIntensityd, RandSpatialCropd, RandFlipd, ToTensord,
    ConvertToMultiChannelBasedOnBratsClassesd, CastToTyped, DivisiblePadd
)
import torch
import numpy as np

def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        
        # 1. Tworzymy 3 kanały BraTS (to one dają Bool)
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        
        # 2. KLUCZOWY KROK: Zamieniamy Bool na Float32
        CastToTyped(keys=["image", "label"], dtype=(np.float32, np.float32)),
        
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[128, 128, 128],
            random_center=True,
            random_size=False
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ToTensord(keys=["image", "label"]),
    ])

def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        
        # Tutaj też rzutujemy na float, ze względu na błąd w fazie validacji
        CastToTyped(keys=["image", "label"], dtype=(np.float32, np.float32)),

        # Unikanie 
        DivisiblePadd(keys=["image", "label"], k=16),
        
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])