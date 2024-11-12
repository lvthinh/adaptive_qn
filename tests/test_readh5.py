import h5py

def read_h5_file(file_path):
    """
    Reads the contents of an HDF5 (.h5) file and displays the datasets within it.

    Args:
        file_path (str): The path to the .h5 file.

    Returns:
        None
    """
    # Open the .h5 file in read mode
    with h5py.File(file_path, 'r') as f:
        print(f"Reading file: {file_path}")
        
        # Recursively print the structure of the HDF5 file
        def print_h5_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"\nDataset: {name}")
                print(f"Shape: {obj.shape}, Data Type: {obj.dtype}")
                # Display part of the dataset (if it is large, limit printing)
                print(f"Data (first 5 entries):\n{obj[...][:]}")
            elif isinstance(obj, h5py.Group):
                print(f"\nGroup: {name}")

        # Walk through the file structure and print datasets and groups
        f.visititems(print_h5_structure)

# Example usage: Change 'your_file.h5' to the path of your HDF5 file.
read_h5_file('C:/Users/tvle2/Documents/Code/queso/data/estimates.h5')
