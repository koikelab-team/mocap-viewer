"""
Render T-pose SMPLX model with EMG-based coloring to video.
Reads EMG data and colors body parts with red intensity based on EMG values.
"""
import os
# Suppress OpenMP duplicate library warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import pickle
import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.headless import HeadlessRenderer
from aitviewer.utils.utils import images_to_video

# Load configuration from capviewer/data/aitvconfig.yaml if it exists
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONFIG_PATH = os.path.join(SCRIPT_DIR, "data", "aitvconfig.yaml")
if os.path.exists(LOCAL_CONFIG_PATH):
    print(f"Loading configuration from: {LOCAL_CONFIG_PATH}")
    C.update_conf(LOCAL_CONFIG_PATH)

# Verify SMPLX models path exists
if not os.path.exists(C.smplx_models):
    raise FileNotFoundError(
        f"SMPLX models path does not exist: {C.smplx_models}\n"
        f"Please check your configuration file at: {LOCAL_CONFIG_PATH}"
    )
print(f"Using SMPLX models from: {C.smplx_models}")

# EMG data path (relative to script location)
EMG_PATH = os.path.join(SCRIPT_DIR, "data", "emg", "01_20250725_TaichiSugiyama_01.pkl")
SEGMENTATION_PATH = os.path.join(SCRIPT_DIR, "data", "smplx_vert_segmentation.json")
OUTPUT_VIDEO_PATH = os.path.join(SCRIPT_DIR, "..", "outputs", "emg_tpose_video.mp4")

# Video resolution
VIDEO_WIDTH = 256
VIDEO_HEIGHT = 256

# Downsampling factor for time axis
DOWNSAMPLE_FACTOR = 5

# EMG channel to body part mapping
EMG_CHANNELS = {
    0: 'rightShoulder',
    1: 'leftShoulder',
    2: 'rightArm',
    3: 'leftArm',
    4: 'spine1',
    5: 'spine2',
    6: 'rightUpLeg',
    7: 'leftUpLeg',
}


def load_emg_data(emg_path):
    """Load EMG data from pkl file."""
    print(f"Loading EMG data from: {emg_path}")
    with open(emg_path, 'rb') as f:
        emg_data = pickle.load(f, encoding='latin1')
    
    if isinstance(emg_data, np.ndarray):
        print(f"EMG data shape: {emg_data.shape}")
        return emg_data
    else:
        raise ValueError(f"Unexpected EMG data format: {type(emg_data)}")


def load_segmentation(seg_path):
    """Load body part segmentation from JSON file."""
    print(f"Loading segmentation from: {seg_path}")
    with open(seg_path, 'r', encoding='utf-8') as f:
        seg_data = json.load(f)
    return seg_data


def normalize_emg_values(emg_data):
    """
    Normalize EMG values to 0-1 range for each channel.
    Uses min-max normalization per channel.
    """
    normalized = np.zeros_like(emg_data)
    for ch in range(emg_data.shape[1]):
        ch_data = emg_data[:, ch]
        ch_min = np.min(ch_data)
        ch_max = np.max(ch_data)
        if ch_max > ch_min:
            normalized[:, ch] = (ch_data - ch_min) / (ch_max - ch_min)
        else:
            normalized[:, ch] = 0.0
    return normalized


def create_colored_tpose(emg_data, seg_data, num_frames):
    """
    Create T-pose SMPLX sequence with EMG-based coloring.
    
    Args:
        emg_data: (num_frames, 8) EMG values
        seg_data: Dictionary mapping body part names to vertex indices
        num_frames: Number of frames
    
    Returns:
        SMPLSequence with colored vertices
    """
    # Create SMPLX layer
    smpl_layer = SMPLLayer(
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        device=C.device,
    )
    
    # Create T-pose sequence with specified number of frames
    print(f"Creating T-pose sequence with {num_frames} frames...")
    poses_body = np.zeros([num_frames, smpl_layer.bm.NUM_BODY_JOINTS * 3])
    seq = SMPLSequence(
        poses_body=poses_body,
        smpl_layer=smpl_layer,
        name="EMG T-Pose",
    )
    
    # Get vertex indices for each body part
    part_vertex_indices = {}
    for ch_idx, part_name in EMG_CHANNELS.items():
        if part_name in seg_data:
            part_vertex_indices[ch_idx] = np.array(seg_data[part_name], dtype=np.int32)
            print(f"  {part_name}: {len(part_vertex_indices[ch_idx])} vertices")
        else:
            print(f"  Warning: {part_name} not found in segmentation!")
            part_vertex_indices[ch_idx] = np.array([], dtype=np.int32)
    
    # Normalize EMG values to 0-1 range
    emg_normalized = normalize_emg_values(emg_data)
    
    # Get number of vertices
    n_vertices = seq.vertices.shape[1]
    
    # Create vertex colors array: (num_frames, n_vertices, 4) RGBA
    # Start with white base color
    vertex_colors = np.ones((num_frames, n_vertices, 4), dtype=np.float32)
    vertex_colors[:, :, :3] = 1.0  # White RGB
    
    # Apply red coloring based on EMG values
    print("Applying EMG-based coloring...")
    for frame_idx in range(num_frames):
        for ch_idx, vertex_indices in part_vertex_indices.items():
            if len(vertex_indices) == 0:
                continue
            
            # Get normalized EMG value for this channel and frame
            emg_value = emg_normalized[frame_idx, ch_idx]
            
            # Apply red color: white (1,1,1) -> red (1,0,0) based on EMG value
            # Higher EMG = more red (higher intensity)
            # Interpolate: white (1,1,1) when emg=0, red (1,0,0) when emg=1
            vertex_colors[frame_idx, vertex_indices, 0] = 1.0  # Keep R at 1.0 (full red)
            vertex_colors[frame_idx, vertex_indices, 1] = 1.0 - emg_value  # Reduce G (0 when emg=1)
            vertex_colors[frame_idx, vertex_indices, 2] = 1.0 - emg_value  # Reduce B (0 when emg=1)
            # Alpha stays at 1.0
    
    # Set vertex colors (shape: n_frames, n_vertices, 4)
    print(f"Setting vertex colors: shape {vertex_colors.shape}")
    seq.mesh_seq.vertex_colors = vertex_colors
    print("Vertex colors applied!")
    
    return seq


def main():
    """Main function to render EMG-colored T-pose video."""
    # Load EMG data
    emg_data = load_emg_data(EMG_PATH)
    original_num_frames = emg_data.shape[0]
    
    # Downsample EMG data by taking every DOWNSAMPLE_FACTOR-th frame
    print(f"Original frames: {original_num_frames}")
    print(f"Downsampling by factor {DOWNSAMPLE_FACTOR}...")
    emg_data = emg_data[::DOWNSAMPLE_FACTOR]
    num_frames = emg_data.shape[0]
    print(f"Downsampled frames: {num_frames}")
    
    # Load segmentation
    seg_data = load_segmentation(SEGMENTATION_PATH)
    
    # Create colored T-pose sequence
    seq = create_colored_tpose(emg_data, seg_data, num_frames)
    
    # Create headless renderer with specified resolution
    print(f"Creating renderer with resolution {VIDEO_WIDTH}x{VIDEO_HEIGHT}...")
    renderer = HeadlessRenderer(size=(VIDEO_WIDTH, VIDEO_HEIGHT))
    
    # Disable auto_set_floor and auto_set_camera_target to avoid NumPy 2.0 compatibility issue
    renderer.auto_set_floor = False
    renderer.auto_set_camera_target = False
    renderer.scene.floor.enabled = False
    
    # Set black background
    renderer.scene.background_color = (0.0, 0.0, 0.0, 1.0)  # Black background
    
    # Enable shadows for better model visibility
    renderer.shadows_enabled = True
    
    # Adjust lighting for better contrast
    # Reduce ambient light to make model stand out more
    renderer.scene.ambient_strength = 0.3  # Lower ambient = more contrast
    
    # Adjust light positions for better visibility
    if len(renderer.scene.lights) >= 2:
        # Front light - main illumination
        renderer.scene.lights[1].position = (0.0, 15.0, 15.0) if not C.z_up else (0.0, 15.0, -15.0)
        renderer.scene.lights[1].light_color = (1.0, 1.0, 1.0)  # White light
        # Back light - rim lighting for edge definition
        renderer.scene.lights[0].position = (0.0, -10.0, 10.0) if not C.z_up else (0.0, 10.0, -10.0)
        renderer.scene.lights[0].light_color = (0.8, 0.8, 0.8)  # Slightly dimmer
    
    renderer.scene.add(seq)
    
    # Hide coordinate axes (the blue lines)
    renderer.scene.origin.enabled = False
    
    # Set camera to front view with downward angle
    # For Y-up coordinate system: camera in front (positive Z), elevated, looking down
    # Distance: 2.5 units away from model
    camera_distance = 2.5
    camera_height = 0.5  # Higher elevation for more downward angle
    if C.z_up:
        # Z-up: camera in front (negative Y direction), elevated in Z
        renderer.scene.camera.position = np.array([0.0, -camera_distance, camera_height])
        # Target below center for downward angle
        renderer.scene.camera.target = np.array([0.0, 0.0, -0.4])
    else:
        # Y-up: camera in front (positive Z direction), elevated in Y
        renderer.scene.camera.position = np.array([0.0, camera_height, camera_distance])
        # Target below center for downward angle
        renderer.scene.camera.target = np.array([0.0, -0.4, 0.0])
    
    # Set up vector (Y-up or Z-up depending on coordinate system)
    if C.z_up:
        renderer.scene.camera.up = np.array([0.0, 0.0, 1.0])
    else:
        renderer.scene.camera.up = np.array([0.0, 1.0, 0.0])
    
    print("Camera set to front view")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary frame directory
    frame_dir = OUTPUT_VIDEO_PATH.replace(".mp4", "_frames")
    if os.path.exists(frame_dir):
        import shutil
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    
    # Initialize scene
    renderer._init_scene()
    
    # Render frames manually (avoiding skvideo DLL issue)
    print(f"Rendering {num_frames} frames to: {frame_dir}")
    print("This may take a while...")
    
    from tqdm import tqdm
    output_fps = 30.0
    dt = 1.0 / output_fps
    time = 0.0
    
    # Setup animation
    saved_run_animations = renderer.run_animations
    saved_curr_frame = renderer.scene.current_frame_id
    renderer.run_animations = True
    renderer.scene.current_frame_id = 0
    renderer._last_frame_rendered_at = 0
    
    # Render each frame
    for i in tqdm(range(num_frames), desc="Rendering frames"):
        renderer.render(time, time + dt, export=True, transparent_background=False)
        img = renderer.get_current_frame_as_image(alpha=False)
        
        # Save frame
        img_name = os.path.join(frame_dir, "frame_{:06d}.png".format(i))
        img.save(img_name)
        
        # Advance to next frame
        renderer.scene.current_frame_id = i
        time += dt
    
    # Restore viewer state
    renderer.run_animations = saved_run_animations
    renderer.scene.current_frame_id = saved_curr_frame
    
    # Convert frames to video using ffmpeg
    print(f"Converting frames to video: {OUTPUT_VIDEO_PATH}")
    images_to_video(
        frame_dir=frame_dir,
        video_path=OUTPUT_VIDEO_PATH,
        frame_format="frame_%06d.png",
        input_fps=output_fps,
        output_fps=output_fps,
        start_frame=0,
    )
    
    # Clean up frame directory
    print("Cleaning up temporary frames...")
    import shutil
    shutil.rmtree(frame_dir)
    
    print(f"Video saved to: {os.path.abspath(OUTPUT_VIDEO_PATH)}")


if __name__ == "__main__":
    main()

