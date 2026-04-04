import os
import pandas as pd
from sklearn.model_selection import train_test_split
from monai.data import DataLoader, PersistentDataset
from src.transforms import get_train_transforms, get_val_transforms

def get_brats_loaders(csv_path, batch_size=2):
    """
    Reads the inventory CSV and prepares MONAI DataLoaders with persistent caching.
    """
    df = pd.read_csv(csv_path)
    
    data_dicts = []
    for _, row in df.iterrows():
        data_dicts.append({
            # Pass a list of paths - LoadImaged automatically stacks them into channels (T1n, T1c, T2w, T2f)
            "image": [row['t1n'], row['t1c'], row['t2w'], row['t2f']],
            # Get from 'seg' column, but label it 'label' in the dictionary
            "label": row['seg'] 
        })

    # Split into training and validation sets (80% train, 20% validation)
    train_files, val_files = train_test_split(data_dicts, test_size=0.2, random_state=42)

    # Define the persistent cache directory to store preprocessed 3D volumes
    cache_dir = os.path.join(os.getcwd(), "monai_persistent_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Use PersistentDataset to save processed tensors to disk, significantly reducing CPU bottleneck
    train_ds = PersistentDataset(data=train_files, transform=get_train_transforms(), cache_dir=cache_dir)
    val_ds = PersistentDataset(data=val_files, transform=get_val_transforms(), cache_dir=cache_dir)

    # Initialize DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader