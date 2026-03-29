from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd,
    Orientationd, NormalizeIntensityd, RandSpatialCropd,
    RandFlipd, ToTensord
)

def get_train_transforms():
    """
    Transforms for training: includes random cropping and flipping (augmentation).
    """
    return Compose([
        # 1. Load volumes from paths provided in the dictionary
        LoadImaged(keys=["t1n", "t1c", "t2w", "t2f", "label"]), 
        
        # 2. Ensure channel dimension is first (C, H, W, D)
        EnsureChannelFirstd(keys=["t1n", "t1c", "t2w", "t2f", "label"]),
        
        # 3. Combine 4 MRI modalities into a single 4-channel 'image' tensor
        ConcatItemsd(keys=["t1n", "t1c", "t2w", "t2f"], name="image"),
        
        # 4. Standardize orientation to RAS
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # 5. Z-score intensity normalization per channel
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        
        # 6. Randomly crop a 128x128x128 cube
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[128, 128, 128],
            random_size=False
        ),
        
        # 7. Data augmentation: Random horizontal flip
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        
        # 8. Convert to PyTorch Tensors
        ToTensord(keys=["image", "label"]),
    ])

def get_val_transforms():
    """
    Transforms for validation: No random augmentations, no cropping.
    """
    return Compose([
        LoadImaged(keys=["t1n", "t1c", "t2w", "t2f", "label"]), 
        EnsureChannelFirstd(keys=["t1n", "t1c", "t2w", "t2f", "label"]),
        ConcatItemsd(keys=["t1n", "t1c", "t2w", "t2f"], name="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])