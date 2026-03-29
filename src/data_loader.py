import pandas as pd
from sklearn.model_selection import train_test_split
from monai.data import CacheDataset, DataLoader
from .transforms import get_train_transforms, get_val_transforms

def get_brats_loaders(csv_path, batch_size=2, cache_rate=1.0):
    """
    Creates MONAI DataLoaders for training and validation.
    """
    # 1. Load the inventory CSV
    df = pd.read_csv(csv_path)
    
    # 2. Map CSV columns to dictionary keys (Crucial: mapping 'seg' to 'label')
    data_dicts = [
        {
            "t1n": row['t1n'], 
            "t1c": row['t1c'], 
            "t2w": row['t2w'], 
            "t2f": row['t2f'], 
            "label": row['seg'] # Mapping here ensures 'label' key exists
        }
        for _, row in df.iterrows()
    ]

    # 3. Split into 80% Train / 20% Val
    train_files, val_files = train_test_split(data_dicts, test_size=0.2, random_state=42)

    # 4. Create Datasets with Caching for speed
    train_ds = CacheDataset(
        data=train_files, 
        transform=get_train_transforms(), 
        cache_rate=cache_rate, 
        num_workers=4
    )
    val_ds = CacheDataset(
        data=val_files, 
        transform=get_val_transforms(), 
        cache_rate=cache_rate, 
        num_workers=4
    )

    # 5. Final DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader