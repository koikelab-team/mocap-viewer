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
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.renderables.billboard import Billboard


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


def extract_frames_from_video(video_path, max_frames=None):
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None for all)
        
    Returns:
        List of frames as numpy arrays (RGB format)
    """
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        return []
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1
        
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames


def create_single_camera_with_video(viewer, video_path, distance=5.0, camera_position=None):
    """
    Create a single camera with video billboard.
    
    Args:
        viewer: The viewer instance
        video_path: Path to video file
        distance: Distance from camera for billboard
        camera_position: Camera position [x, y, z] (None for default)
        
    Returns:
        Tuple of (camera, billboard)
    """
    # Extract frames from video
    frames = extract_frames_from_video(video_path)
    if not frames:
        print("Error: No frames extracted from video")
        return None, None
    
    # Get video dimensions from first frame
    first_frame = frames[0]
    rows, cols = first_frame.shape[:2]
    
    # Default camera position
    if camera_position is None:
        camera_position = np.array([5.0, 1.5, 5.0], dtype=np.float32)
    
    # Camera looks at origin
    target = np.array([0.0, 0.0, 0.0])
    position = np.array(camera_position, dtype=np.float32)
    
    # Compute camera extrinsics (OpenCV format)
    # Forward vector (points from camera to target, normalized)
    forward = target - position
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    
    # World up vector
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    # Right vector (perpendicular to forward and world_up)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        # If forward is parallel to world_up, use a different up
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
    right = right / (right_norm + 1e-8)
    
    # Down vector (negative of up, perpendicular to right and forward)
    down = -np.cross(right, forward)
    down = down / (np.linalg.norm(down) + 1e-8)
    
    # OpenCV camera: Rt = [R|t] where:
    # - R rows are: right, down, forward (in world coordinates)
    # - t = -R @ position
    R = np.array([
        right,
        down,
        forward
    ], dtype=np.float32)
    
    # Translation: t = -R @ position
    t = -R @ position
    
    # Extrinsics matrix [R|t]
    Rt = np.hstack([R, t.reshape(3, 1)]).astype(np.float32)
    
    # Intrinsics (simple pinhole camera)
    fov = 60  # degrees
    f = max(cols, rows) / 2.0 / np.tan(np.radians(fov / 2))
    K = np.array([
        [f, 0, cols / 2],
        [0, f, rows / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create OpenCV camera with n_frames matching video
    num_frames = len(frames)
    
    # Expand K and Rt to match number of frames
    K_expanded = np.repeat(K[np.newaxis, ...], num_frames, axis=0)
    Rt_expanded = np.repeat(Rt[np.newaxis, ...], num_frames, axis=0)
    
    camera = OpenCVCamera(
        K=K_expanded,
        Rt=Rt_expanded,
        cols=cols,
        rows=rows,
        viewer=viewer,
        name="Camera"
    )
    
    # Show frustum (camera viewing range)
    try:
        frustum_distance = camera.far if camera.far is not None else distance
        camera.show_frustum(cols, rows, frustum_distance)
        print(f"  Enabled frustum for Camera (distance: {frustum_distance:.2f})")
    except Exception as e:
        print(f"  Warning: Could not show frustum for Camera: {e}")
    
    # Create billboard from camera and video frames
    # Convert frames list to numpy array for Billboard
    # Billboard expects frames in shape (N, H, W, C) where C is channels (RGB)
    frames_array = np.array(frames)  # Shape: (N, H, W, 3)
    
    print(f"  Creating billboard with {num_frames} frames at distance {distance}")
    billboard = Billboard.from_camera_and_distance(
        camera,
        distance,
        cols,
        rows,
        frames_array,  # Pass frames as numpy array
    )
    
    return camera, billboard


def create_example_cameras(viewer, num_cameras=4, distance=5.0):
    """
    Create example cameras around the scene for visualization.
    
    Args:
        viewer: The viewer instance
        num_cameras: Number of cameras to create
        distance: Distance from origin for cameras
        
    Returns:
        List of created cameras
    """
    cameras = []
    cols, rows = 1920, 1080
    
    # Create cameras in a circle around the origin
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        
        # Camera position
        x = distance * np.cos(angle)
        y = 1.5  # Height
        z = distance * np.sin(angle)
        
        # Camera looks at origin
        target = np.array([0.0, 0.0, 0.0])
        position = np.array([x, y, z], dtype=np.float32)
        
        # Compute camera extrinsics (OpenCV format)
        # Forward vector (points from camera to target, normalized)
        forward = target - position
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # World up vector
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Right vector (perpendicular to forward and world_up)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            # If forward is parallel to world_up, use a different up
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)
        right = right / (right_norm + 1e-8)
        
        # Down vector (negative of up, perpendicular to right and forward)
        down = -np.cross(right, forward)
        down = down / (np.linalg.norm(down) + 1e-8)
        
        # OpenCV camera: Rt = [R|t] where:
        # - R rows are: right, down, forward (in world coordinates)
        # - t = -R @ position
        R = np.array([
            right,
            down,
            forward
        ], dtype=np.float32)
        
        # Translation: t = -R @ position
        t = -R @ position
        
        # Extrinsics matrix [R|t]
        Rt = np.hstack([R, t.reshape(3, 1)]).astype(np.float32)
        
        # Intrinsics (simple pinhole camera)
        fov = 60  # degrees
        f = max(cols, rows) / 2.0 / np.tan(np.radians(fov / 2))
        K = np.array([
            [f, 0, cols / 2],
            [0, f, rows / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Create OpenCV camera
        camera = OpenCVCamera(
            K=K,
            Rt=Rt,
            cols=cols,
            rows=rows,
            viewer=viewer,
            name=f"Camera {i}"
        )
        
        # Show frustum (camera viewing range)
        try:
            # Use far plane distance as frustum distance, or default to 10.0
            frustum_distance = camera.far if camera.far is not None else 10.0
            camera.show_frustum(cols, rows, frustum_distance)
            print(f"  Enabled frustum for Camera {i} (distance: {frustum_distance:.2f})")
        except Exception as e:
            print(f"  Warning: Could not show frustum for Camera {i}: {e}")
        
        cameras.append(camera)
        viewer.scene.add(camera)
        print(f"Created example camera {i} at position ({x:.2f}, {y:.2f}, {z:.2f}) with frustum")
    
    return cameras


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
    parser.add_argument(
        "--camera-info",
        type=str,
        default=None,
        help="Path to camera info .npz file (with intrinsics, extrinsics, etc.) (optional)"
    )
    parser.add_argument(
        "--camera-images",
        type=str,
        default=None,
        help="Path to directory containing camera images (optional)"
    )
    parser.add_argument(
        "--create-example-cameras",
        action="store_true",
        help="Create example cameras around the scene if no camera info is provided"
    )
    parser.add_argument(
        "--camera-video",
        type=str,
        default=None,
        help="Path to camera video file (e.g., MP4) to display on billboard"
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
        camera_info_path=args.camera_info,
        camera_images_path=args.camera_images,
    )
    
    # Create single camera with video if provided
    if args.camera_video:
        if not os.path.exists(args.camera_video):
            print(f"Error: Camera video file not found: {args.camera_video}")
            sys.exit(1)
        
        print(f"Creating camera with video: {args.camera_video}")
        camera, billboard = create_single_camera_with_video(
            viewer, 
            args.camera_video, 
            distance=5.0
        )
        
        if camera is not None and billboard is not None:
            viewer.scene.add(camera, billboard)
            # Add to viewer's camera list for GUI
            viewer.cameras.append(camera)
            viewer.show_cameras[0] = True
            viewer.camera_textures[0] = None
            # Enable camera GUI
            if "cameras" not in viewer.gui_controls:
                viewer.gui_controls["cameras"] = viewer.gui_cameras
            if "camera_settings" not in viewer.gui_controls:
                viewer.gui_controls["camera_settings"] = viewer.gui_camera_settings
            print("Camera and billboard added to scene")
        else:
            print("Error: Failed to create camera with video")
    
    # Create example cameras if requested and no camera info provided
    elif args.create_example_cameras and not args.camera_info:
        print("Creating example cameras...")
        example_cameras = create_example_cameras(viewer, num_cameras=4, distance=5.0)
        # Add cameras to viewer's camera list
        viewer.cameras.extend(example_cameras)
        for i in range(len(example_cameras)):
            idx = len(viewer.cameras) - len(example_cameras) + i
            viewer.show_cameras[idx] = True
            viewer.camera_textures[idx] = None
        # Enable camera GUI
        if "cameras" not in viewer.gui_controls:
            viewer.gui_controls["cameras"] = viewer.gui_cameras
    
    viewer.run_animations = True
    viewer.playback_fps = 30.0
    
    try:
        viewer.run()
    finally:
        viewer.release()

