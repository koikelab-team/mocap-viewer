"""
Test script for CAP Viewer.

This script tests the CAP Viewer with a sample SMPLX sequence.
For testing without video files, it will still work - just won't show video panels.
"""
import os
import sys

# Add parent directory to path to import convert_animation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capviewer import CapViewer


def main():
    # Test with a sample SMPLX file
    # You can adjust this path to your actual data
    smplx_path = "tmp/generated_motion_denorm.npz"
    
    # Check if file exists
    if not os.path.exists(smplx_path):
        print(f"Error: SMPLX file not found: {smplx_path}")
        print("\nTrying to find alternative files...")
        
        # Try to find any .npz file in common directories
        search_dirs = ["tmp", "checkdata", "results_1111"]
        found = False
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith('.npz') and 'motion' in file.lower():
                            smplx_path = os.path.join(root, file)
                            print(f"Found: {smplx_path}")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        if not found:
            print("\nPlease provide a valid SMPLX animation file (.npz format)")
            print("You can:")
            print("  1. Place a .npz file in the tmp/ directory")
            print("  2. Or modify this script to point to your file")
            print("\nExample usage:")
            print("  python capviewer/test_capviewer.py")
            print("  python capviewer/capviewer.py <path_to_smplx.npz> --foot-pressure <video.mp4> --emg <video.mp4>")
            return
    
    print(f"Loading SMPLX from: {smplx_path}")
    
    # Optional: Add video paths if you have them
    foot_pressure_path = None  # Set this if you have a foot pressure video
    emg_path = None  # Set this if you have an EMG video
    
    # Example: Uncomment and set paths if you have videos
    # foot_pressure_path = "path/to/foot_pressure.mp4"
    # emg_path = "path/to/emg_signal.mp4"
    
    # Create viewer
    print("\nStarting CAP Viewer...")
    print("Controls:")
    print("  - Main window: 3D SMPLX animation")
    print("  - Floating panels: Foot Pressure and EMG videos (if provided)")
    print("  - All views are synchronized to the same frame")
    print("  - Use the Playback panel to control animation")
    print("  - Drag panels to reposition them")
    print("\nPress ESC to exit")
    
    viewer = CapViewer(
        smplx_path=smplx_path,
        foot_pressure_path=foot_pressure_path,
        emg_path=emg_path
    )
    
    viewer.run_animations = True
    viewer.playback_fps = 30.0
    
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\nViewer interrupted by user")
    finally:
        viewer.release()
        print("Viewer closed.")


if __name__ == "__main__":
    main()



