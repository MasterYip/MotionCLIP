import numpy as np
import os

def profile_npz_file(file_path):
    """Load and print the profile of an NPZ file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        data = np.load(file_path)
        print(f"File: {file_path}")
        print(f"Keys: {list(data.keys())}")
        print("\nData details:")
        for key in data.keys():
            array = data[key]
            print(f"  {key}:")
            print(f"    Shape: {array.shape}")
            print(f"    Dtype: {array.dtype}")
            if np.issubdtype(array.dtype, np.number):
                print(f"    Min: {array.min()}, Max: {array.max()}")
            else:
                print(f"    Sample data: {array.flatten()[:]}")
        data.close()
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    # Example: profile a single NPZ file
    npz_file = "MotionCLIP/data/amass/BioMotionLab_NTroje/rub002/0000_treadmill_norm_poses_120_jpos.npz"
    profile_npz_file(npz_file)
    
    # Or process multiple NPZ files in a directory
    # npz_dir = "/path/to/npz/directory"
    # for file in os.listdir(npz_dir):
    #     if file.endswith(".npz"):
    #         profile_npz_file(os.path.join(npz_dir, file))