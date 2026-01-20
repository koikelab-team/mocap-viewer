"""
View the dummy SMPLX motion sequence using CapViewer.
If --foot-pressure is a pkl file, it will be automatically converted to MP4.
"""
import os
import sys
import argparse
import tempfile
import cv2
import numpy as np
import pickle
from PIL import Image, ImageDraw

# Add parent directory to path
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from capviewer import CapViewer


def viridis_colormap(value):
    """
    Convert a value [0, 1] to RGB color using viridis colormap approximation.
    
    Args:
        value: float between 0 and 1
        
    Returns:
        Tuple of (R, G, B) values in [0, 255]
    """
    # Simplified viridis colormap approximation
    value = np.clip(value, 0.0, 1.0)
    
    if value < 0.25:
        # Dark blue to blue
        t = value / 0.25
        r = int(68 * t)
        g = int(1 * t)
        b = int(84 + (159 - 84) * t)
    elif value < 0.5:
        # Blue to green
        t = (value - 0.25) / 0.25
        r = int(68 + (53 - 68) * t)
        g = int(1 + (42 - 1) * t)
        b = int(159 + (135 - 159) * t)
    elif value < 0.75:
        # Green to yellow
        t = (value - 0.5) / 0.25
        r = int(53 + (253 - 53) * t)
        g = int(42 + (231 - 42) * t)
        b = int(135 + (37 - 135) * t)
    else:
        # Yellow to bright yellow
        t = (value - 0.75) / 0.25
        r = int(253 + (254 - 253) * t)
        g = int(231 + (237 - 231) * t)
        b = int(37 + (34 - 37) * t)
    
    return (r, g, b)


def load_pkl_data(pkl_path):
    """
    Load numpy array data from pkl file.
    
    Args:
        pkl_path: Path to the pkl file
        
    Returns:
        numpy array containing the insole pressure data
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    return data


def generate_insole_mp4_from_pkl(pkl_path, output_path=None, fps=10):
    """
    Generate MP4 video from insole pressure pkl file.
    
    Args:
        pkl_path: Path to the pkl file containing insole pressure data
        output_path: Path to save the output MP4 (if None, uses temp file)
        fps: Frames per second for the video
        
    Returns:
        Path to the generated MP4 file
    """
    # Load data from pkl file
    print(f"Loading insole pressure data from: {pkl_path}")
    insole_data = load_pkl_data(pkl_path)
    
    # Normalize insole data to [0, 1] range for visualization
    if insole_data.max() > 1.0 or insole_data.min() < 0.0:
        insole_data = (insole_data - insole_data.min()) / (insole_data.max() - insole_data.min() + 1e-8)
    
    print(f'Insole pressure data shape: {insole_data.shape}')
    
    # Determine output path
    if output_path is None:
        # Create temp file
        temp_dir = tempfile.gettempdir()
        base_name = os.path.splitext(os.path.basename(pkl_path))[0]
        output_path = os.path.join(temp_dir, f"{base_name}_insole_animation.mp4")
    
    # Load the insole background
    script_dir = os.path.dirname(os.path.abspath(__file__))
    insole_image_path = os.path.join(script_dir, 'data', 'insole.png')
    
    if not os.path.exists(insole_image_path):
        print(f"Warning: Insole image not found at {insole_image_path}")
        print("Skipping insole heatmap visualization.")
        return None
    
    insole_image = Image.open(insole_image_path)
    
    # Adjusted sensor positions based on the image dimensions (296x666)
    sensor_positions = [
        (159, 598), (95, 604), (163, 512), (89, 515),
        (167, 419), (89, 428), (212, 308), (62, 334),
        (240, 182), (187, 183), (145, 187), (105, 197),
        (61, 207), (223, 71), (167, 79), (102, 92)
    ]
    
    # Create a composite image
    width, height = insole_image.size
    video_width = width * 2
    video_height = height
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
    
    if not out.isOpened():
        print(f"Error: Failed to create video writer for {output_path}")
        return None
    
    # Process each frame
    print(f"Generating MP4 video with {len(insole_data)} frames...")
    for frame in range(len(insole_data)):
        composite_image = Image.new('RGB', (video_width, video_height))
        img = insole_image.copy().convert('RGB')
        
        # Draw left foot sensors
        draw = ImageDraw.Draw(img)
        for i, (x, y) in enumerate(sensor_positions):
            radius = 20
            color = viridis_colormap(insole_data[frame, i+16])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        composite_image.paste(img, (0, 0))

        # Draw right foot sensors
        img_right = img.copy()
        draw = ImageDraw.Draw(img_right)
        for i, (x, y) in enumerate(sensor_positions):
            radius = 20
            color = viridis_colormap(insole_data[frame, i])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        composite_image.paste(img_right.transpose(Image.FLIP_LEFT_RIGHT), (width, 0))

        # Convert PIL image to OpenCV format (BGR)
        frame_array = np.array(composite_image)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        out.write(frame_bgr)
    
    # Release video writer
    out.release()
    print(f"Insole heatmap MP4 saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View dummy SMPLX motion with optional Foot Pressure & EMG")
    parser.add_argument(
        "--pkl-path",
        type=str,
        default="capviewer/data/smplx/dummy_motion_800.pkl",
        help="Path to dummy SMPLX motion pkl file (default: capviewer/data/smplx/dummy_motion_800.pkl)"
    )
    parser.add_argument(
        "--foot-pressure",
        type=str,
        default=None,
        help="Path to foot pressure video file or pkl file (optional). If pkl, will be converted to MP4."
    )
    parser.add_argument(
        "--emg",
        type=str,
        default=None,
        help="Path to EMG signal video file (optional)"
    )
    
    args = parser.parse_args()
    
    # Check if pkl file exists
    if not os.path.exists(args.pkl_path):
        print(f"Error: File not found: {args.pkl_path}")
        print("Please run 'python capviewer/generate_dummy_smplx.py' first to generate the dummy data.")
        sys.exit(1)
    
    print(f"Loading dummy SMPLX motion from: {args.pkl_path}")
    
    # Handle foot pressure path
    foot_pressure_path = args.foot_pressure
    if args.foot_pressure:
        # Check if it's a pkl file
        if args.foot_pressure.lower().endswith('.pkl'):
            if not os.path.exists(args.foot_pressure):
                print(f"Error: Foot pressure pkl file not found: {args.foot_pressure}")
                sys.exit(1)
            # Generate MP4 from pkl
            foot_pressure_path = generate_insole_mp4_from_pkl(args.foot_pressure)
            if foot_pressure_path is None:
                print("Error: Failed to generate MP4 from pkl file.")
                sys.exit(1)
        else:
            # It's a video file
            if not os.path.exists(args.foot_pressure):
                print(f"Error: Foot pressure video file not found: {args.foot_pressure}")
                sys.exit(1)
            print(f"Foot pressure video: {args.foot_pressure}")
    
    if args.emg:
        print(f"EMG video: {args.emg}")
    
    print("Starting CAP Viewer...")
    
    # Create viewer
    viewer = CapViewer(
        smplx_path=args.pkl_path,
        foot_pressure_path=foot_pressure_path,
        emg_path=args.emg,
    )
    
    viewer.run_animations = True
    viewer.playback_fps = 30.0
    
    try:
        viewer.run()
    finally:
        viewer.release()

