"""
Parser to convert JSON files from data/temp/output/smpl/smpl/xxxxxx.json 
to PKL format matching capviewer/data/smplx/dummy_motion_800.pkl
"""
import json
import pickle
import numpy as np
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def swap_axes_translation(trans, axis_swap='yz'):
    """
    Swap axes for translation vector.
    
    Args:
        trans: Translation vector (3,) or (F, 3)
        axis_swap: Which axes to swap. Options: 'yz', 'xz', 'xy', 'zyx'
        
    Returns:
        Swapped translation vector
    """
    trans = np.array(trans)
    original_shape = trans.shape
    if len(trans.shape) == 1:
        trans = trans.reshape(1, -1)
    
    if axis_swap == 'yz':
        # Swap Y and Z: (x, y, z) -> (x, z, y)
        trans_swapped = trans.copy()
        trans_swapped[:, [1, 2]] = trans_swapped[:, [2, 1]]
    elif axis_swap == 'xz':
        # Swap X and Z: (x, y, z) -> (z, y, x)
        trans_swapped = trans.copy()
        trans_swapped[:, [0, 2]] = trans_swapped[:, [2, 0]]
    elif axis_swap == 'xy':
        # Swap X and Y: (x, y, z) -> (y, x, z)
        trans_swapped = trans.copy()
        trans_swapped[:, [0, 1]] = trans_swapped[:, [1, 0]]
    elif axis_swap == 'zyx':
        # Full swap: (x, y, z) -> (z, y, x) then negate Y
        trans_swapped = trans.copy()
        trans_swapped[:, [0, 2]] = trans_swapped[:, [2, 0]]
        trans_swapped[:, 1] = -trans_swapped[:, 1]
    else:
        trans_swapped = trans
    
    if len(original_shape) == 1:
        return trans_swapped[0]
    return trans_swapped


def apply_rotation_matrix_to_aa(rot_aa, rot_matrix):
    """
    Apply a rotation matrix to axis-angle rotations.
    
    Args:
        rot_aa: Rotation vectors (F, 3) in axis-angle format
        rot_matrix: 3x3 rotation matrix to apply
        
    Returns:
        Transformed rotation vectors in axis-angle format
    """
    rot_aa = np.array(rot_aa)
    original_shape = rot_aa.shape
    if len(rot_aa.shape) == 1:
        rot_aa = rot_aa.reshape(1, -1)
    
    # Convert axis-angle to rotation matrices
    rotations = R.from_rotvec(rot_aa)
    rot_mats = rotations.as_matrix()  # (F, 3, 3)
    
    # Apply coordinate system transformation
    # R_new = R_transform @ R_old
    rot_matrix = np.array(rot_matrix)
    transformed_rots = rot_matrix @ rot_mats  # (F, 3, 3)
    
    # Convert back to axis-angle
    transformed_aa = R.from_matrix(transformed_rots).as_rotvec()
    
    if len(original_shape) == 1:
        return transformed_aa[0]
    return transformed_aa


def apply_rotation_matrix_to_translation(trans, rot_matrix):
    """
    Apply a rotation matrix to translation vectors.
    
    Args:
        trans: Translation vectors (F, 3)
        rot_matrix: 3x3 rotation matrix to apply
        
    Returns:
        Transformed translation vectors
    """
    trans = np.array(trans)
    original_shape = trans.shape
    if len(trans.shape) == 1:
        trans = trans.reshape(1, -1)
    
    # Apply rotation matrix to translation vectors
    rot_matrix = np.array(rot_matrix)
    # For translation, we apply the rotation: t_new = R @ t_old
    transformed_trans = (rot_matrix @ trans.T).T  # (F, 3)
    
    if len(original_shape) == 1:
        return transformed_trans[0]
    return transformed_trans


def swap_axes_rotation(rot, axis_swap='yz'):
    """
    Swap axes for rotation vector (axis-angle representation).
    
    Args:
        rot: Rotation vector (3,) or (F, 3) in axis-angle format
        axis_swap: Which axes to swap/transform. Options: 'yz', 'xz', 'xy', 'neg_y', 'neg_z', 'neg_x', 
                   'rot_x_90', 'rot_x_-90', 'rot_y_90', 'rot_y_-90', 'rot_z_90', 'rot_z_-90'
        
    Returns:
        Swapped rotation vector
    """
    rot = np.array(rot)
    original_shape = rot.shape
    if len(rot.shape) == 1:
        rot = rot.reshape(1, -1)
    
    # Handle rotation matrix transformations
    if axis_swap == 'rot_x_90':
        # Rotate 90 degrees around X axis
        rot_matrix = R.from_euler('x', 90, degrees=True).as_matrix()
        return apply_rotation_matrix_to_aa(rot, rot_matrix)
    elif axis_swap == 'rot_x_-90':
        # Rotate -90 degrees around X axis
        rot_matrix = R.from_euler('x', -90, degrees=True).as_matrix()
        return apply_rotation_matrix_to_aa(rot, rot_matrix)
    elif axis_swap == 'rot_y_90':
        # Rotate 90 degrees around Y axis
        rot_matrix = R.from_euler('y', 90, degrees=True).as_matrix()
        return apply_rotation_matrix_to_aa(rot, rot_matrix)
    elif axis_swap == 'rot_y_-90':
        # Rotate -90 degrees around Y axis
        rot_matrix = R.from_euler('y', -90, degrees=True).as_matrix()
        return apply_rotation_matrix_to_aa(rot, rot_matrix)
    elif axis_swap == 'rot_z_90':
        # Rotate 90 degrees around Z axis
        rot_matrix = R.from_euler('z', 90, degrees=True).as_matrix()
        return apply_rotation_matrix_to_aa(rot, rot_matrix)
    elif axis_swap == 'rot_z_-90':
        # Rotate -90 degrees around Z axis
        rot_matrix = R.from_euler('z', -90, degrees=True).as_matrix()
        return apply_rotation_matrix_to_aa(rot, rot_matrix)
    elif axis_swap == 'yz':
        # Swap Y and Z rotation axes
        rot_swapped = rot.copy()
        rot_swapped[:, [1, 2]] = rot_swapped[:, [2, 1]]
    elif axis_swap == 'xz':
        # Swap X and Z rotation axes
        rot_swapped = rot.copy()
        rot_swapped[:, [0, 2]] = rot_swapped[:, [2, 0]]
    elif axis_swap == 'xy':
        # Swap X and Y rotation axes
        rot_swapped = rot.copy()
        rot_swapped[:, [0, 1]] = rot_swapped[:, [1, 0]]
    elif axis_swap == 'neg_y':
        # Negate Y axis
        rot_swapped = rot.copy()
        rot_swapped[:, 1] = -rot_swapped[:, 1]
    elif axis_swap == 'neg_z':
        # Negate Z axis
        rot_swapped = rot.copy()
        rot_swapped[:, 2] = -rot_swapped[:, 2]
    elif axis_swap == 'neg_x':
        # Negate X axis
        rot_swapped = rot.copy()
        rot_swapped[:, 0] = -rot_swapped[:, 0]
    elif axis_swap == 'yz_neg_y':
        # Swap Y and Z, then negate Y
        rot_swapped = rot.copy()
        rot_swapped[:, [1, 2]] = rot_swapped[:, [2, 1]]
        rot_swapped[:, 1] = -rot_swapped[:, 1]
    elif axis_swap == 'yz_neg_z':
        # Swap Y and Z, then negate Z
        rot_swapped = rot.copy()
        rot_swapped[:, [1, 2]] = rot_swapped[:, [2, 1]]
        rot_swapped[:, 2] = -rot_swapped[:, 2]
    else:
        rot_swapped = rot
    
    if len(original_shape) == 1:
        return rot_swapped[0]
    return rot_swapped


def load_json_file(json_path):
    """
    Load JSON file and return data.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary containing JSON data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def convert_json_to_pkl_format(json_data):
    """
    Convert JSON data to PKL format matching dummy_motion_800.pkl.
    
    Expected PKL format:
    - poses_body: (F, 63) float32 - 21 body joints * 3 (axis-angle)
    - poses_root: (F, 3) float32 - root rotation (axis-angle)
    - betas: (10,) float32 - shape parameters
    - trans: (F, 3) float32 - root translation
    - num_frames: int - number of frames
    
    Args:
        json_data: Dictionary containing JSON data
        
    Returns:
        Dictionary in PKL format
    """
    # Try different possible JSON formats
    poses_body = None
    poses_root = None
    trans = None
    betas = None
    num_frames = None
    
    # Method 1: Check if data is a list of frames
    if isinstance(json_data, list):
        num_frames = len(json_data)
        # Try to extract pose data from each frame
        if num_frames > 0 and isinstance(json_data[0], dict):
            # Extract poses_body (21 joints * 3 = 63)
            poses_body_list = []
            poses_root_list = []
            trans_list = []
            
            for frame in json_data:
                # Try different key names for body pose
                body_pose = None
                if 'body_pose' in frame:
                    body_pose = np.array(frame['body_pose'], dtype=np.float32)
                elif 'poses_body' in frame:
                    body_pose = np.array(frame['poses_body'], dtype=np.float32)
                elif 'poses' in frame:
                    # Handle nested list format: [[0.0, 0.0, ...]]
                    pose_data = np.array(frame['poses'], dtype=np.float32)
                    # Flatten nested arrays
                    if len(pose_data.shape) > 1:
                        pose_data = pose_data.flatten()
                    
                    # Handle different pose formats
                    if len(pose_data) == 87:
                        # SMPLX format: first 3 are global_orient (skip), next 63 are body pose
                        body_pose = pose_data[3:66]  # Skip first 3, take next 63
                    elif len(pose_data) == 75:
                        # SMPLX format variant: first 3 are global_orient (skip), next 63 are body pose
                        body_pose = pose_data[3:66]  # Skip first 3, take next 63
                    elif len(pose_data) >= 63:
                        # Standard format: first 63 are body pose
                        body_pose = pose_data[:63]
                    elif len(pose_data) > 0:
                        body_pose = pose_data
                elif 'pose' in frame:
                    pose_data = np.array(frame['pose'], dtype=np.float32)
                    if len(pose_data.shape) > 1:
                        pose_data = pose_data.flatten()
                    if len(pose_data) >= 63:
                        body_pose = pose_data[:63]
                    elif len(pose_data) > 0:
                        body_pose = pose_data
                
                if body_pose is not None:
                    # Reshape to (63,) if needed
                    if len(body_pose.shape) > 1:
                        body_pose = body_pose.flatten()
                    # Ensure 63 elements (pad or truncate)
                    if len(body_pose) >= 63:
                        poses_body_list.append(body_pose[:63])
                    else:
                        # Pad with zeros if less than 63
                        padded = np.zeros(63, dtype=np.float32)
                        padded[:len(body_pose)] = body_pose
                        poses_body_list.append(padded)
                
                # Extract root rotation (global_orient)
                root_pose = None
                if 'global_orient' in frame:
                    root_pose = np.array(frame['global_orient'], dtype=np.float32)
                elif 'poses_root' in frame:
                    root_pose = np.array(frame['poses_root'], dtype=np.float32)
                elif 'root_pose' in frame:
                    root_pose = np.array(frame['root_pose'], dtype=np.float32)
                elif 'Rh' in frame:
                    # Handle nested list format: [[0.877, -1.703, -1.62]]
                    root_data = np.array(frame['Rh'], dtype=np.float32)
                    if len(root_data.shape) > 1:
                        root_pose = root_data.flatten()
                    else:
                        root_pose = root_data
                
                if root_pose is not None:
                    if len(root_pose.shape) > 1:
                        root_pose = root_pose.flatten()
                    poses_root_list.append(root_pose[:3])  # Ensure 3 elements
                
                # Extract translation
                translation = None
                if 'translation' in frame:
                    translation = np.array(frame['translation'], dtype=np.float32)
                elif 'trans' in frame:
                    translation = np.array(frame['trans'], dtype=np.float32)
                elif 'Th' in frame:
                    # Handle nested list format: [[0.375, 0.339, 1.086]]
                    trans_data = np.array(frame['Th'], dtype=np.float32)
                    if len(trans_data.shape) > 1:
                        translation = trans_data.flatten()
                    else:
                        translation = trans_data
                
                if translation is not None:
                    if len(translation.shape) > 1:
                        translation = translation.flatten()
                    trans_list.append(translation[:3])  # Ensure 3 elements
            
            # Convert to numpy arrays
            if poses_body_list:
                poses_body = np.array(poses_body_list, dtype=np.float32)
            if poses_root_list:
                poses_root = np.array(poses_root_list, dtype=np.float32)
            if trans_list:
                trans = np.array(trans_list, dtype=np.float32)
    
    # Method 2: Check if data is a dictionary with frame arrays
    elif isinstance(json_data, dict):
        # Check for array-based format
        if 'body_pose' in json_data or 'poses_body' in json_data:
            body_key = 'body_pose' if 'body_pose' in json_data else 'poses_body'
            body_data = np.array(json_data[body_key], dtype=np.float32)
            
            # Reshape if needed: (F, 21, 3) -> (F, 63) or (F, 63) -> (F, 63)
            if len(body_data.shape) == 3 and body_data.shape[1] == 21 and body_data.shape[2] == 3:
                poses_body = body_data.reshape(body_data.shape[0], -1)
            elif len(body_data.shape) == 2:
                poses_body = body_data
            else:
                poses_body = body_data.reshape(-1, 63)
            
            num_frames = poses_body.shape[0]
        
        # Extract root rotation
        if 'global_orient' in json_data:
            root_data = np.array(json_data['global_orient'], dtype=np.float32)
            if len(root_data.shape) == 2:
                poses_root = root_data
            else:
                poses_root = root_data.reshape(-1, 3)
        elif 'poses_root' in json_data:
            root_data = np.array(json_data['poses_root'], dtype=np.float32)
            if len(root_data.shape) == 2:
                poses_root = root_data
            else:
                poses_root = root_data.reshape(-1, 3)
        
        # Extract translation
        if 'translation' in json_data:
            trans_data = np.array(json_data['translation'], dtype=np.float32)
            if len(trans_data.shape) == 2:
                trans = trans_data
            else:
                trans = trans_data.reshape(-1, 3)
        elif 'trans' in json_data:
            trans_data = np.array(json_data['trans'], dtype=np.float32)
            if len(trans_data.shape) == 2:
                trans = trans_data
            else:
                trans = trans_data.reshape(-1, 3)
        
        # Extract betas
        if 'betas' in json_data:
            beta_data = np.array(json_data['betas'], dtype=np.float32)
            if len(beta_data.shape) > 1:
                beta_data = beta_data.flatten()
            betas = beta_data[:10] if len(beta_data) >= 10 else np.pad(beta_data, (0, 10 - len(beta_data)), 'constant')
        
        # Update num_frames if not set
        if num_frames is None:
            if poses_body is not None:
                num_frames = poses_body.shape[0]
            elif poses_root is not None:
                num_frames = poses_root.shape[0]
            elif trans is not None:
                num_frames = trans.shape[0]
    
    # Create default values if missing
    if poses_body is None:
        raise ValueError("Could not extract poses_body from JSON. Please check the JSON format.")
    
    if num_frames is None:
        num_frames = poses_body.shape[0]
    
    # Ensure poses_body has correct shape (F, 63)
    if poses_body.shape[1] != 63:
        if poses_body.shape[1] > 63:
            poses_body = poses_body[:, :63]
        else:
            # Pad with zeros if needed
            padding = np.zeros((num_frames, 63 - poses_body.shape[1]), dtype=np.float32)
            poses_body = np.concatenate([poses_body, padding], axis=1)
    
    # Create default poses_root if missing
    if poses_root is None:
        poses_root = np.zeros((num_frames, 3), dtype=np.float32)
    elif poses_root.shape[0] != num_frames:
        # Pad or truncate to match num_frames
        if poses_root.shape[0] < num_frames:
            padding = np.zeros((num_frames - poses_root.shape[0], 3), dtype=np.float32)
            poses_root = np.concatenate([poses_root, padding], axis=0)
        else:
            poses_root = poses_root[:num_frames]
    
    # Create default trans if missing
    if trans is None:
        trans = np.zeros((num_frames, 3), dtype=np.float32)
    elif trans.shape[0] != num_frames:
        # Pad or truncate to match num_frames
        if trans.shape[0] < num_frames:
            padding = np.zeros((num_frames - trans.shape[0], 3), dtype=np.float32)
            trans = np.concatenate([trans, padding], axis=0)
        else:
            trans = trans[:num_frames]
    
    # Create default betas if missing
    if betas is None:
        betas = np.zeros(10, dtype=np.float32)
    
    # Ensure betas has correct shape (10,)
    if len(betas) != 10:
        if len(betas) > 10:
            betas = betas[:10]
        else:
            betas = np.pad(betas, (0, 10 - len(betas)), 'constant')
    
    # Package data in PKL format
    pkl_data = {
        'poses_body': poses_body.astype(np.float32),  # (F, 63)
        'poses_root': poses_root.astype(np.float32),  # (F, 3)
        'betas': betas.astype(np.float32),            # (10,)
        'trans': trans.astype(np.float32),            # (F, 3)
        'num_frames': int(num_frames),
    }
    
    return pkl_data


def convert_json_to_pkl(json_path, output_path=None):
    """
    Convert a single JSON file to PKL format.
    
    Args:
        json_path: Path to input JSON file
        output_path: Path to output PKL file (if None, auto-generates)
        
    Returns:
        Path to output PKL file
    """
    print(f"Converting: {json_path}")
    
    # Load JSON data
    json_data = load_json_file(json_path)
    
    # Convert to PKL format
    pkl_data = convert_json_to_pkl_format(json_data)
    
    # Generate output path if not provided
    if output_path is None:
        json_stem = Path(json_path).stem
        output_dir = Path(json_path).parent
        output_path = output_dir / f"{json_stem}.pkl"
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_dir = output_path.parent
    if output_dir and str(output_dir) != '.' and str(output_dir) != '':
        os.makedirs(output_dir, exist_ok=True)
    
    # Save PKL file
    with open(output_path, 'wb') as f:
        pickle.dump(pkl_data, f)
    
    print(f"  Saved to: {output_path}")
    print(f"  Shapes: poses_body={pkl_data['poses_body'].shape}, "
          f"poses_root={pkl_data['poses_root'].shape}, "
          f"betas={pkl_data['betas'].shape}, "
          f"trans={pkl_data['trans'].shape}, "
          f"num_frames={pkl_data['num_frames']}")
    
    return output_path


def convert_all_json_to_single_pkl(input_dir, output_path=None, axis_swap=None):
    """
    Convert all JSON files in input directory to a single PKL file.
    Each JSON file represents one frame, and all frames are combined into one sequence.
    
    Args:
        input_dir: Directory containing JSON files
        output_path: Path to output PKL file (if None, auto-generates)
        axis_swap: Axis swap mode for coordinate conversion. Options: 'yz', 'xz', 'xy', 'zyx', or None
        
    Returns:
        Path to output PKL file
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all JSON files and sort by filename
    json_files = sorted(list(input_path.glob("*.json")))
    
    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in: {input_dir}")
    
    print(f"Found {len(json_files)} JSON file(s), merging into single sequence...")
    
    # Collect all frames
    all_poses_body = []
    all_poses_root = []
    all_trans = []
    betas = None
    
    for idx, json_file in enumerate(json_files):
        try:
            # Load JSON data
            json_data = load_json_file(json_file)
            
            # Extract frame data
            if isinstance(json_data, list) and len(json_data) > 0:
                frame = json_data[0]  # Each JSON file contains one frame in a list
            elif isinstance(json_data, dict):
                frame = json_data
            else:
                print(f"Warning: Skipping {json_file} - unexpected format")
                continue
            
            # Extract body pose
            body_pose = None
            if 'poses' in frame:
                pose_data = np.array(frame['poses'], dtype=np.float32)
                if len(pose_data.shape) > 1:
                    pose_data = pose_data.flatten()
                
                # Handle different pose formats
                if len(pose_data) == 87:
                    # SMPLX format: first 3 are global_orient (skip), next 63 are body pose
                    body_pose = pose_data[3:66]  # Skip first 3, take next 63
                elif len(pose_data) == 75:
                    # SMPLX format variant: first 3 are global_orient (skip), next 63 are body pose, then 9 more
                    body_pose = pose_data[3:66]  # Skip first 3, take next 63
                elif len(pose_data) >= 63:
                    # Standard format: first 63 are body pose
                    body_pose = pose_data[:63]
                elif len(pose_data) > 0:
                    body_pose = pose_data
            
            if body_pose is not None:
                if len(body_pose.shape) > 1:
                    body_pose = body_pose.flatten()
                if len(body_pose) >= 63:
                    all_poses_body.append(body_pose[:63])
                else:
                    padded = np.zeros(63, dtype=np.float32)
                    padded[:len(body_pose)] = body_pose
                    all_poses_body.append(padded)
            else:
                # Use zeros if not found
                all_poses_body.append(np.zeros(63, dtype=np.float32))
            
            # Extract root rotation
            root_pose = None
            if 'Rh' in frame:
                root_data = np.array(frame['Rh'], dtype=np.float32)
                if len(root_data.shape) > 1:
                    root_pose = root_data.flatten()
                else:
                    root_pose = root_data
            elif 'global_orient' in frame:
                root_data = np.array(frame['global_orient'], dtype=np.float32)
                if len(root_data.shape) > 1:
                    root_pose = root_data.flatten()
                else:
                    root_pose = root_data
            
            if root_pose is not None:
                if len(root_pose.shape) > 1:
                    root_pose = root_pose.flatten()
                all_poses_root.append(root_pose[:3])
            else:
                all_poses_root.append(np.zeros(3, dtype=np.float32))
            
            # Extract translation
            translation = None
            if 'Th' in frame:
                trans_data = np.array(frame['Th'], dtype=np.float32)
                if len(trans_data.shape) > 1:
                    translation = trans_data.flatten()
                else:
                    translation = trans_data
            elif 'translation' in frame:
                trans_data = np.array(frame['translation'], dtype=np.float32)
                if len(trans_data.shape) > 1:
                    translation = trans_data.flatten()
                else:
                    translation = trans_data
            
            if translation is not None:
                if len(translation.shape) > 1:
                    translation = translation.flatten()
                all_trans.append(translation[:3])
            else:
                all_trans.append(np.zeros(3, dtype=np.float32))
            
            # Extract betas (use from first frame if available)
            if betas is None:
                beta_data = None
                if 'betas' in frame:
                    beta_data = np.array(frame['betas'], dtype=np.float32)
                elif 'shapes' in frame:
                    # SMPLX uses 'shapes' instead of 'betas'
                    beta_data = np.array(frame['shapes'], dtype=np.float32)
                
                if beta_data is not None:
                    if len(beta_data.shape) > 1:
                        beta_data = beta_data.flatten()
                    if len(beta_data) >= 10:
                        betas = beta_data[:10]
                    else:
                        betas = np.pad(beta_data, (0, 10 - len(beta_data)), 'constant')
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(json_files)} files...")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_poses_body) == 0:
        raise ValueError("No valid frames extracted from JSON files")
    
    # Convert to numpy arrays
    num_frames = len(all_poses_body)
    poses_body = np.array(all_poses_body, dtype=np.float32)  # (F, 63)
    poses_root = np.array(all_poses_root, dtype=np.float32)  # (F, 3)
    trans = np.array(all_trans, dtype=np.float32)  # (F, 3)
    
    # Apply axis swap if specified (for coordinate system conversion)
    if axis_swap is not None:
        print(f"Applying axis swap: {axis_swap}")
        
        # Check if this is a rotation matrix transformation
        if axis_swap.startswith('rot_'):
            # For rotation matrix transformations, apply to both rotation and translation
            if axis_swap == 'rot_x_90':
                rot_matrix = R.from_euler('x', 90, degrees=True).as_matrix()
            elif axis_swap == 'rot_x_-90':
                rot_matrix = R.from_euler('x', -90, degrees=True).as_matrix()
            elif axis_swap == 'rot_y_90':
                rot_matrix = R.from_euler('y', 90, degrees=True).as_matrix()
            elif axis_swap == 'rot_y_-90':
                rot_matrix = R.from_euler('y', -90, degrees=True).as_matrix()
            elif axis_swap == 'rot_z_90':
                rot_matrix = R.from_euler('z', 90, degrees=True).as_matrix()
            elif axis_swap == 'rot_z_-90':
                rot_matrix = R.from_euler('z', -90, degrees=True).as_matrix()
            else:
                rot_matrix = None
            
            if rot_matrix is not None:
                # Apply rotation matrix to both rotation and translation
                poses_root = apply_rotation_matrix_to_aa(poses_root, rot_matrix)
                trans = apply_rotation_matrix_to_translation(trans, rot_matrix)
        else:
            # For simple axis swaps, only apply to rotation (translation handled separately if needed)
            poses_root = swap_axes_rotation(poses_root, axis_swap)
    
    # Use default betas if not found
    if betas is None:
        betas = np.zeros(10, dtype=np.float32)
    else:
        betas = betas.astype(np.float32)
    
    # Package data in PKL format
    pkl_data = {
        'poses_body': poses_body,  # (F, 63)
        'poses_root': poses_root,  # (F, 3)
        'betas': betas,            # (10,)
        'trans': trans,            # (F, 3)
        'num_frames': int(num_frames),
    }
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.name}_merged.pkl"
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_dir = output_path.parent
    if output_dir and str(output_dir) != '.' and str(output_dir) != '':
        os.makedirs(output_dir, exist_ok=True)
    
    # Save PKL file
    with open(output_path, 'wb') as f:
        pickle.dump(pkl_data, f)
    
    print(f"\nSuccessfully merged {num_frames} frames into: {output_path}")
    print(f"  Shapes: poses_body={poses_body.shape}, "
          f"poses_root={poses_root.shape}, "
          f"betas={betas.shape}, "
          f"trans={trans.shape}, "
          f"num_frames={num_frames}")
    
    return output_path


def convert_all_json_files(input_dir, output_dir=None):
    """
    Convert all JSON files in input directory to individual PKL files.
    (Legacy function - kept for backward compatibility)
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Output directory (if None, saves in same directory)
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Find all JSON files
    json_files = list(input_path.glob("*.json"))
    
    if len(json_files) == 0:
        print(f"No JSON files found in: {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON file(s)")
    
    # Convert each file
    for json_file in json_files:
        try:
            if output_dir is None:
                output_path = None  # Auto-generate in same directory
            else:
                output_path = Path(output_dir) / f"{json_file.stem}.pkl"
            
            convert_json_to_pkl(json_file, output_path)
        except Exception as e:
            print(f"Error converting {json_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert JSON files to PKL format")
    parser.add_argument("input", type=str, help="Input JSON file or directory")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output PKL file path")
    parser.add_argument("--separate", action="store_true", help="Convert each JSON file separately (default: merge all into one)")
    parser.add_argument("--axis-swap", type=str, default=None, 
                       choices=['yz', 'xz', 'xy', 'zyx', 'neg_y', 'neg_z', 'neg_x', 'yz_neg_y', 'yz_neg_z',
                                'rot_x_90', 'rot_x_-90', 'rot_y_90', 'rot_y_-90', 'rot_z_90', 'rot_z_-90'],
                       help="Transform rotation axes (translation unchanged). Options include axis swaps and rotations (rot_x_90, rot_y_90, etc.)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Convert single file
        convert_json_to_pkl(input_path, args.output)
    elif input_path.is_dir():
        if args.separate:
            # Convert all JSON files separately (legacy mode)
            convert_all_json_files(input_path, args.output)
        else:
            # Merge all JSON files into one PKL file (default)
            convert_all_json_to_single_pkl(input_path, args.output, axis_swap=args.axis_swap)
    else:
        print(f"Error: Input path does not exist: {args.input}")

