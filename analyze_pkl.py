"""
Script to analyze pickle files and show their structure.
"""
import pickle
import numpy as np
import sys
import os

def analyze_pkl(file_path):
    """Analyze a pickle file and print its structure."""
    print(f"Analyzing: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type: {type(data)}")
        print()
        
        if isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys:")
            print()
            for key, value in data.items():
                print(f"  Key: '{key}'")
                print(f"    Type: {type(value)}")
                
                if isinstance(value, np.ndarray):
                    print(f"    Shape: {value.shape}")
                    print(f"    Dtype: {value.dtype}")
                    print(f"    Min: {np.min(value):.6f}, Max: {np.max(value):.6f}")
                    if value.size < 20:
                        print(f"    Values: {value}")
                elif isinstance(value, (list, tuple)):
                    print(f"    Length: {len(value)}")
                    if len(value) > 0:
                        print(f"    First element type: {type(value[0])}")
                        if isinstance(value[0], np.ndarray):
                            print(f"    First element shape: {value[0].shape}")
                elif isinstance(value, (int, float)):
                    print(f"    Value: {value}")
                elif isinstance(value, str):
                    print(f"    Value: {value[:100]}..." if len(value) > 100 else f"    Value: {value}")
                else:
                    print(f"    Value: {value}")
                print()
        
        elif isinstance(data, (list, tuple)):
            print(f"{type(data).__name__} with {len(data)} elements")
            if len(data) > 0:
                print(f"First element type: {type(data[0])}")
                if isinstance(data[0], np.ndarray):
                    print(f"First element shape: {data[0].shape}")
                    print(f"First element dtype: {data[0].dtype}")
        
        elif isinstance(data, np.ndarray):
            print(f"NumPy array")
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Min: {np.min(data):.6f}, Max: {np.max(data):.6f}")
        
        else:
            print(f"Value: {data}")
            print(f"Attributes: {dir(data)}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "capviewer/data/smplx/01_20250725_TaichiSugiyama_01.pkl"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    analyze_pkl(file_path)

