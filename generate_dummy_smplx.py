"""
Generate a dummy 800-frame SMPLX motion sequence and save as pkl.
"""
import pickle
import numpy as np
import os

def generate_dummy_smplx_motion(num_frames=800):
    """
    Generate a dummy SMPLX motion sequence.
    
    Args:
        num_frames: Number of frames (default: 800)
        
    Returns:
        Dictionary with SMPLX motion data
    """
    print(f"Generating dummy SMPLX motion with {num_frames} frames...")
    
    # SMPLX has 21 body joints (excluding root)
    num_body_joints = 21
    
    # Generate poses_body: (F, 21*3) - body joint rotations in axis-angle
    # Create a simple waving motion
    poses_body = np.zeros((num_frames, num_body_joints * 3), dtype=np.float32)
    
    # Add some simple motion to the arms (joints 13-20 are arms)
    for frame in range(num_frames):
        # Simple sine wave for arm motion
        t = frame / num_frames * 2 * np.pi
        
        # Left arm (shoulder, elbow, wrist) - indices in body joints
        # Left shoulder (body joint 15 in SMPL: left_shoulder)
        left_shoulder_idx = 15 * 3
        poses_body[frame, left_shoulder_idx + 1] = 0.5 * np.sin(t)  # Y rotation
        
        # Left elbow (body joint 17: left_elbow)
        left_elbow_idx = 17 * 3
        poses_body[frame, left_elbow_idx + 1] = 0.3 * np.sin(t + np.pi/4)  # Y rotation
        
        # Right arm - similar but opposite phase
        right_shoulder_idx = 16 * 3  # right_shoulder
        poses_body[frame, right_shoulder_idx + 1] = -0.5 * np.sin(t)
        
        right_elbow_idx = 18 * 3  # right_elbow
        poses_body[frame, right_elbow_idx + 1] = -0.3 * np.sin(t + np.pi/4)
        
        # Add some spine rotation
        spine1_idx = 2 * 3  # spine1
        poses_body[frame, spine1_idx + 2] = 0.2 * np.sin(t * 0.5)  # Z rotation
    
    # Generate poses_root: (F, 3) - root rotation in axis-angle
    poses_root = np.zeros((num_frames, 3), dtype=np.float32)
    # Add slight rotation
    for frame in range(num_frames):
        t = frame / num_frames * 2 * np.pi
        poses_root[frame, 2] = 0.1 * np.sin(t * 0.3)  # Z rotation
    
    # Generate trans: (F, 3) - root translation (set to zero, no translation)
    trans = np.zeros((num_frames, 3), dtype=np.float32)
    
    # Generate betas: (10,) - shape parameters (neutral shape)
    betas = np.zeros(10, dtype=np.float32)
    
    # Package data
    data = {
        'poses_body': poses_body,  # (800, 63)
        'poses_root': poses_root,  # (800, 3)
        'betas': betas,            # (10,)
        'trans': trans,            # (800, 3)
        'num_frames': num_frames,
    }
    
    print(f"Generated data shapes:")
    print(f"  poses_body: {poses_body.shape}")
    print(f"  poses_root: {poses_root.shape}")
    print(f"  betas: {betas.shape}")
    print(f"  trans: {trans.shape}")
    
    return data

if __name__ == "__main__":
    # Generate dummy motion
    data = generate_dummy_smplx_motion(800)
    
    # Save as pkl
    output_path = "capviewer/data/smplx/dummy_motion_800.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nSaved dummy SMPLX motion to: {output_path}")

