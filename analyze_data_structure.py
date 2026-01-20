"""
Detailed analysis of the data structure.
"""
import pickle
import numpy as np
import os

def analyze_smplx_data():
    """Analyze SMPLX data structure."""
    file_path = "capviewer/data/smplx/01_20250725_TaichiSugiyama_01.pkl"
    
    print("=" * 80)
    print("SMPLX Data Analysis")
    print("=" * 80)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print()
    
    # Check if it's rotation matrices (4x4) or quaternions
    print("First frame, first joint:")
    print(data[0, 0, :])
    print()
    
    print("First frame, all joints shape:", data[0, :, :].shape)
    print()
    
    # Check if values are in valid rotation range
    print("Value range per dimension:")
    for i in range(4):
        print(f"  Dim {i}: min={np.min(data[:, :, i]):.6f}, max={np.max(data[:, :, i]):.6f}, mean={np.mean(data[:, :, i]):.6f}")
    print()
    
    # Check if it looks like rotation matrices (4x4 would be flattened to 16)
    # But we have 4, so it might be:
    # 1. Quaternions (w, x, y, z)
    # 2. Rotation matrix first row (4 values)
    # 3. Axis-angle representation (3) + something else
    
    # Check if it's quaternions (should have norm close to 1)
    if data.shape[2] == 4:
        norms = np.linalg.norm(data, axis=2)
        print("Quaternion norms (should be ~1.0):")
        print(f"  Min: {np.min(norms):.6f}")
        print(f"  Max: {np.max(norms):.6f}")
        print(f"  Mean: {np.mean(norms):.6f}")
        print(f"  Std: {np.std(norms):.6f}")
        print()
    
    # Sample a few frames
    print("Sample frames:")
    for frame_idx in [0, 100, 200, 399]:
        if frame_idx < data.shape[0]:
            print(f"Frame {frame_idx}, joint 0: {data[frame_idx, 0, :]}")
    print()
    
    print("Possible interpretations:")
    print("  - Shape (800, 21, 4) suggests:")
    print("    * 800 frames")
    print("    * 21 joints (SMPL/SMPLX has 21 body joints)")
    print("    * 4 values per joint:")
    print("      - Could be quaternions (w, x, y, z)")
    print("      - Could be rotation matrix first row")
    print("      - Could be axis-angle (3) + scale (1)")


if __name__ == "__main__":
    analyze_smplx_data()

