"""
Data visualizer script for insole pressure and pose data.
Outputs MP4 videos instead of GIFs.
"""
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle
import os
import sys


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


def plot_insole_heatmap_mp4(insole_image, sensor_values, output_path, fps=10):
    """
    Create an MP4 video showing insole pressure heatmap.
    
    Args:
        insole_image: PIL Image of the insole background
        sensor_values: numpy array of shape (num_frames, num_sensors)
        output_path: Path to save the output MP4
        fps: Frames per second for the video
    """
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
    
    # Process each frame
    for frame in range(len(sensor_values)):
        composite_image = Image.new('RGB', (video_width, video_height))
        img = insole_image.copy().convert('RGB')
        
        # Draw left foot sensors
        draw = ImageDraw.Draw(img)
        for i, (x, y) in enumerate(sensor_positions):
            radius = 20
            color = viridis_colormap(sensor_values[frame, i+16])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        composite_image.paste(img, (0, 0))

        # Draw right foot sensors
        img_right = img.copy()
        draw = ImageDraw.Draw(img_right)
        for i, (x, y) in enumerate(sensor_positions):
            radius = 20
            color = viridis_colormap(sensor_values[frame, i])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        composite_image.paste(img_right.transpose(Image.FLIP_LEFT_RIGHT), (width, 0))

        # Convert PIL image to OpenCV format (BGR)
        frame_array = np.array(composite_image)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        out.write(frame_bgr)
    
    # Release video writer
    out.release()


def main():
    """Main function to run the visualization."""
    # Specify the pkl file path
    pkl_path = r"C:\Users\13097\aitviewer\capviewer\data\insole\01_20250725_TaichiSugiyama_01.pkl"
    
    # Check if file exists
    if not os.path.exists(pkl_path):
        print(f"Error: File not found: {pkl_path}")
        sys.exit(1)
    
    # Load data from pkl file
    print(f"Loading data from: {pkl_path}")
    insole_data = load_pkl_data(pkl_path)
    
    # Normalize insole data to [0, 1] range for visualization
    if insole_data.max() > 1.0 or insole_data.min() < 0.0:
        insole_data = (insole_data - insole_data.min()) / (insole_data.max() - insole_data.min() + 1e-8)
    
    print(f'Insole pressure data shape: {insole_data.shape}')
    
    # Parameters
    num_sensors = insole_data.shape[1]
    num_timesteps = insole_data.shape[0]
    print(f'{num_timesteps} Frames Loaded, Input Insole Channel: {num_sensors}.')
    
    # Load the insole background
    insole_image_path = os.path.join(os.path.dirname(__file__), 'data', 'insole.png')
    if not os.path.exists(insole_image_path):
        print(f"Warning: Insole image not found at {insole_image_path}")
        print("Skipping insole heatmap visualization.")
    else:
        insole_image = Image.open(insole_image_path)
        
        # Specify the output mp4 path
        output_mp4_path = 'insole_animation.mp4'
        
        # Plot the heatmap and save as mp4
        print(f"Creating insole heatmap MP4: {output_mp4_path}")
        plot_insole_heatmap_mp4(insole_image, insole_data, output_mp4_path, fps=10)
        print(f"Insole heatmap MP4 saved to: {output_mp4_path}")


if __name__ == "__main__":
    main()
