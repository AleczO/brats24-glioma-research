import os
import pandas as pd
from glob import glob

def create_data_manifest(data_dir):
    """Creates a list of paths to all modalities for each patient."""

    patient_dirs = sorted(glob(os.path.join(data_dir, "BraTS*")))
    data_list = []

    print(f"Found {len(patient_dirs)} patient folders.")

    for p_dir in patient_dirs:
        p_id = os.path.basename(p_dir)
        
        try:
            item = {
                "patient_id": p_id,
                "t1n": os.path.abspath(glob(os.path.join(p_dir, "*t1n.nii.gz"))[0]),
                "t1c": os.path.abspath(glob(os.path.join(p_dir, "*t1c.nii.gz"))[0]),
                "t2w": os.path.abspath(glob(os.path.join(p_dir, "*t2w.nii.gz"))[0]),
                "t2f": os.path.abspath(glob(os.path.join(p_dir, "*t2f.nii.gz"))[0]),
                "seg": os.path.abspath(glob(os.path.join(p_dir, "*seg.nii.gz"))[0])
            }
            data_list.append(item)
        except IndexError:
            print(f"Warning: Patient {p_id} has incomplete data!")

    df = pd.DataFrame(data_list)
    df.to_csv("data_inventory.csv", index=False)
    print(f"Inventory complete. Complete patients: {len(df)}")
    return df

# Execute manifest creation
create_data_manifest("data/BraTS-GLI/brats2024-brats-gli-trainingdata/training_data1_v2")