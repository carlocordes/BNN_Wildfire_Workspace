from pathlib import Path
import rasterio
import numpy as np

def produce_weight(path_to_ds: Path):
    # Find and sort all files, ignoring .xml metadata
    file_paths = sorted(f for f in path_to_ds.glob("*") if f.suffix.lower() != ".xml")
    
    # Initialize global counters for the entire dataset
    total_dataset_valid_values = 0
    total_dataset_ones = 0
    total_dataset_zeros = 0
    
    print(f"Analyzing {len(file_paths)} files...")
    
    for file_path in file_paths:
        # Open the raster file
        with rasterio.open(file_path) as src:
            # Using masked=True excludes NoData/Nodata pixels from calculations
            data = src.read(1, masked=True)

        # Accumulate metrics into global counters
        total_dataset_valid_values += data.count() 
        total_dataset_ones += (data == 1).sum()
        total_dataset_zeros += (data == 0).sum()

    # --- Final Dataset-Wide Report ---
    print("\n===============================")
    print("   FINAL DATASET-WIDE TOTALS   ")
    print("===============================")
    print(f"Total Valid Pixels (All Images): {total_dataset_valid_values}")
    print(f"Total Count of Ones:             {total_dataset_ones}")
    print(f"Total Count of Zeros:            {total_dataset_zeros}")

    # Calculate global ratio
    if total_dataset_ones > 0:
        global_zero_to_one_ratio = total_dataset_zeros / total_dataset_ones
        print(f"Global Zero-to-One Ratio:        {global_zero_to_one_ratio:.4f}")
    else:
        print("Global Zero-to-One Ratio:        Undefined (No ones found across any dataset)")
    print("===============================\n")

if __name__ == '__main__':
    path_to_target = Path('files', 'data', 'processed', 'target')
    
    # Ensure the path exists before running to avoid errors
    if path_to_target.exists():
        produce_weight(path_to_target)
    else:
        print(f"Error: The directory '{path_to_target}' does not exist.")