"""
CAP Viewer - Review SMPLX animations with synchronized foot pressure and EMG signals.

This viewer displays:
- SMPLX 3D animation (main view)
- Foot pressure video (floating panel)
- EMG signal video (floating panel)

All views are synchronized to the same frame index.
"""
import os
# Suppress OpenMP duplicate library warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import cv2
import numpy as np
import imgui
from typing import Optional, Tuple

# Add parent directory to path to import convert_animation
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import OpenCVCamera

try:
    from convert_animation import convert_to_aitviewer_format
except ImportError:
    # Fallback: define a simple converter if convert_animation is not available
    def convert_to_aitviewer_format(input_path, **kwargs):
        """Fallback converter that loads using SMPLSequence.from_npz"""
        return SMPLSequence.from_npz(input_path)

# PKL loader for simple SMPLX motion data
import pickle


class VideoPlayer:
    """Helper class to manage video playback and frame extraction."""
    
    def __init__(self, video_path: str):
        """
        Initialize video player.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.current_frame_id = 0
        self._frame_cache = {}  # Cache for loaded frames
        
        if os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Loaded video: {video_path}")
                print(f"  Frames: {self.total_frames}, FPS: {self.fps:.2f}, Size: {self.width}x{self.height}")
            else:
                print(f"Warning: Could not open video: {video_path}")
        else:
            print(f"Warning: Video file not found: {video_path}")
    
    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Get frame at specified index.
        
        Args:
            frame_id: Frame index (0-based)
            
        Returns:
            Frame as numpy array (RGB format), or None if unavailable
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        # Clamp frame_id to valid range
        frame_id = max(0, min(frame_id, self.total_frames - 1))
        
        # Check cache first
        if frame_id in self._frame_cache:
            return self._frame_cache[frame_id]
        
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Cache the frame (limit cache size to avoid memory issues)
            if len(self._frame_cache) < 100:
                self._frame_cache[frame_id] = frame_rgb
            return frame_rgb
        
        return None
    
    def set_frame(self, frame_id: int):
        """Set current frame index."""
        self.current_frame_id = max(0, min(frame_id, self.total_frames - 1))
    
    def release(self):
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._frame_cache.clear()


class CapViewer(Viewer):
    """
    Custom viewer for reviewing SMPLX animations with synchronized foot pressure and EMG videos.
    """
    
    def __init__(
        self,
        smplx_path: str,
        foot_pressure_path: Optional[str] = None,
        emg_path: Optional[str] = None,
        camera_info_path: Optional[str] = None,
        camera_images_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize CAP Viewer.
        
        Args:
            smplx_path: Path to SMPLX animation file (.npz)
            foot_pressure_path: Optional path to foot pressure video file
            emg_path: Optional path to EMG signal video file
            camera_info_path: Optional path to camera info .npz file (with intrinsics, extrinsics, etc.)
            camera_images_path: Optional path to directory containing camera images
            **kwargs: Additional arguments passed to Viewer
        """
        super().__init__(title="CAP Viewer - SMPLX with Foot Pressure & EMG", **kwargs)
        
        # Load SMPLX sequence
        print(f"Loading SMPLX animation from: {smplx_path}")
        try:
            # Check if it's a pkl file
            if smplx_path.endswith('.pkl'):
                print("Detected .pkl file, loading SMPLX motion data...")
                self.smplx_seq = self._load_smplx_from_pkl(smplx_path)
            else:
                # Try standard converter
                self.smplx_seq = convert_to_aitviewer_format(smplx_path)
        except Exception as e:
            print(f"Error loading SMPLX: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to t-pose if loading fails
            print("Falling back to T-pose...")
            self.smplx_seq = SMPLSequence.t_pose()
        
        self.scene.add(self.smplx_seq)
        
        # Initialize video players
        self.foot_pressure_player = None
        self.emg_player = None
        
        if foot_pressure_path:
            self.foot_pressure_player = VideoPlayer(foot_pressure_path)
        
        if emg_path:
            self.emg_player = VideoPlayer(emg_path)
        
        # Panel visibility and positions
        self.show_foot_pressure = foot_pressure_path is not None
        self.show_emg = emg_path is not None
        
        # Panel positions (will be set on first use)
        self.foot_pressure_pos = (50, 100)
        self.foot_pressure_size = (400, 300)
        self.emg_pos = (500, 100)
        self.emg_size = (400, 300)
        
        # Textures for displaying video frames in imgui
        self.foot_pressure_texture = None
        self.emg_texture = None
        
        # Camera system
        self.cameras = []
        self.camera_images_path = camera_images_path
        self.camera_textures = {}  # Dict mapping camera index to texture
        self.show_cameras = {}  # Dict mapping camera index to visibility flag
        self.show_camera_frustums = True  # Global flag for showing camera frustums
        
        # Load cameras if provided
        if camera_info_path and os.path.exists(camera_info_path):
            self._load_cameras(camera_info_path)
        
        # Add custom GUI controls
        self.gui_controls["foot_pressure"] = self.gui_foot_pressure
        self.gui_controls["emg"] = self.gui_emg
        if self.cameras:
            self.gui_controls["cameras"] = self.gui_cameras
            self.gui_controls["camera_settings"] = self.gui_camera_settings
    
    def _load_smplx_from_pkl(self, pkl_path):
        """
        Load SMPLX motion sequence from pkl file.
        
        Args:
            pkl_path: Path to pkl file containing SMPLX motion data
            
        Returns:
            SMPLSequence object
        """
        print(f"Loading SMPLX data from pkl: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract data
        poses_body = data['poses_body']  # (F, 21*3)
        poses_root = data.get('poses_root', None)  # (F, 3) or None
        betas = data.get('betas', None)  # (10,) or None
        trans = data.get('trans', None)  # (F, 3) or None
        
        print(f"Loaded data shapes:")
        print(f"  poses_body: {poses_body.shape}")
        if poses_root is not None:
            print(f"  poses_root: {poses_root.shape}")
        if betas is not None:
            print(f"  betas: {betas.shape}")
        if trans is not None:
            print(f"  trans: {trans.shape}")
        
        # Get SMPL layer from t_pose() to avoid chumpy dependency
        # This reuses an already initialized model
        print("Getting SMPL layer from t_pose()...")
        try:
            temp_seq = SMPLSequence.t_pose()
            smpl_layer = temp_seq.smpl_layer
            print("Successfully obtained SMPL layer from t_pose()")
        except Exception as e:
            print(f"Warning: Failed to get SMPL layer from t_pose(): {e}")
            # Fallback: try to create directly (may fail if chumpy is missing)
            from aitviewer.models.smpl import SMPLLayer
            from aitviewer.configuration import CONFIG as C
            try:
                smpl_layer = SMPLLayer(
                    model_type="smpl",  # Use SMPL for now (21 body joints)
                    gender="neutral",
                    num_betas=10,
                    device=C.device,
                )
                print("Successfully created SMPL layer directly")
            except Exception as e2:
                print(f"Error: Failed to create SMPL layer: {e2}")
                raise
        
        # Create SMPL sequence
        seq = SMPLSequence(
            poses_body=poses_body,
            poses_root=poses_root,
            betas=betas,
            trans=trans,
            smpl_layer=smpl_layer,
            name="SMPLX from PKL",
        )
        
        print("SMPLSequence created successfully!")
        return seq
    
    def _load_cameras(self, camera_info_path: str):
        """
        Load OpenCV cameras from camera info file.
        
        Args:
            camera_info_path: Path to .npz file with camera intrinsics, extrinsics, etc.
        """
        print(f"Loading cameras from: {camera_info_path}")
        try:
            camera_info = np.load(camera_info_path)
            
            # Extract camera data
            ids = camera_info.get("ids", None)
            intrinsics = camera_info.get("intrinsics", None)
            extrinsics = camera_info.get("extrinsics", None)
            dist_coeffs = camera_info.get("dist_coeffs", None)
            
            if intrinsics is None or extrinsics is None:
                print("Warning: Camera info file missing intrinsics or extrinsics")
                return
            
            # Ensure proper shape
            if len(intrinsics.shape) == 2:
                intrinsics = intrinsics[np.newaxis, ...]
            if len(extrinsics.shape) == 2:
                extrinsics = extrinsics[np.newaxis, ...]
            
            # Get image dimensions (default to 1920x1080 if not specified)
            cols = camera_info.get("cols", 1920)
            rows = camera_info.get("rows", 1080)
            
            num_cameras = intrinsics.shape[0]
            print(f"Loading {num_cameras} cameras...")
            
            for i in range(num_cameras):
                cam_id = ids[i] if ids is not None else i
                K = intrinsics[i]
                Rt = extrinsics[i]
                dist = dist_coeffs[i] if dist_coeffs is not None else None
                
                # Create OpenCV camera
                camera = OpenCVCamera(
                    K=K,
                    Rt=Rt,
                    cols=cols,
                    rows=rows,
                    dist_coeffs=dist,
                    viewer=self,
                    name=f"Camera {cam_id}"
                )
                
                # Show frustum (camera viewing range)
                try:
                    # Use far plane distance as frustum distance, or default to 10.0
                    frustum_distance = camera.far if camera.far is not None else 10.0
                    camera.show_frustum(cols, rows, frustum_distance)
                    print(f"  Enabled frustum for Camera {cam_id} (distance: {frustum_distance:.2f})")
                except Exception as e:
                    print(f"  Warning: Could not show frustum for Camera {cam_id}: {e}")
                
                self.cameras.append(camera)
                self.scene.add(camera)
                self.show_cameras[i] = True
                self.camera_textures[i] = None
                
                print(f"  Added Camera {cam_id}")
            
            print(f"Successfully loaded {len(self.cameras)} cameras")
            
        except Exception as e:
            print(f"Error loading cameras: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_camera_image(self, camera_index: int, frame_id: int) -> Optional[np.ndarray]:
        """
        Load image for a specific camera and frame.
        
        Args:
            camera_index: Index of the camera
            frame_id: Frame index
            
        Returns:
            Image as numpy array (RGB format), or None if unavailable
        """
        if not self.camera_images_path or camera_index >= len(self.cameras):
            return None
        
        try:
            camera = self.cameras[camera_index]
            cam_id = camera.name.split()[-1] if "Camera" in camera.name else str(camera_index)
            
            # Try to find image file
            camera_dir = os.path.join(self.camera_images_path, str(cam_id))
            if not os.path.isdir(camera_dir):
                # Try with camera index
                camera_dir = os.path.join(self.camera_images_path, str(camera_index))
            
            if not os.path.isdir(camera_dir):
                return None
            
            # Look for image files (common extensions)
            import glob
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(glob.glob(os.path.join(camera_dir, ext)))
            
            if not image_files:
                return None
            
            # Sort by filename
            image_files.sort()
            
            # Map frame_id to image file
            if frame_id >= len(image_files):
                frame_id = len(image_files) - 1
            
            image_path = image_files[frame_id]
            img = cv2.imread(image_path)
            
            if img is not None:
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img_rgb
            
        except Exception as e:
            print(f"Error loading camera image: {e}")
        
        return None
    
    def gui_foot_pressure(self):
        """GUI panel for foot pressure video."""
        if not self.show_foot_pressure or self.foot_pressure_player is None:
            return
        
        # Set window position and size
        imgui.set_next_window_position(
            self.foot_pressure_pos[0],
            self.foot_pressure_pos[1],
            imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_size(
            self.foot_pressure_size[0],
            self.foot_pressure_size[1],
            imgui.FIRST_USE_EVER
        )
        
        # Create window
        expanded, self.show_foot_pressure = imgui.begin(
            "Foot Pressure##foot_pressure",
            self.show_foot_pressure
        )
        
        if expanded:
            # Get current frame from SMPLX sequence
            current_frame = self.scene.current_frame_id
            
            # Map frame to video frame (handle different frame counts)
            video_frame_id = self._map_frame_to_video(
                current_frame,
                self.scene.n_frames,
                self.foot_pressure_player.total_frames
            )
            
            # Get video frame
            frame = self.foot_pressure_player.get_frame(video_frame_id)
            
            if frame is not None:
                # Update texture if frame changed
                if (self.foot_pressure_texture is None or 
                    self.foot_pressure_player.current_frame_id != video_frame_id):
                    self._update_video_texture(frame, "foot_pressure")
                    self.foot_pressure_player.current_frame_id = video_frame_id
                
                # Display frame info
                imgui.text(f"Frame: {video_frame_id}/{self.foot_pressure_player.total_frames - 1}")
                imgui.text(f"SMPLX Frame: {current_frame}/{self.scene.n_frames - 1}")
                
                # Display video frame
                if self.foot_pressure_texture is not None:
                    # Get available width
                    avail_width = imgui.get_content_region_available()[0]
                    
                    # Calculate display size maintaining aspect ratio
                    aspect_ratio = self.foot_pressure_player.width / max(1, self.foot_pressure_player.height)
                    display_height = min(avail_width / aspect_ratio, 250)
                    display_width = display_height * aspect_ratio
                    
                    # Display image using texture's OpenGL ID
                    # imgui.image expects the OpenGL texture ID
                    imgui.image(
                        self.foot_pressure_texture.glo,
                        display_width,
                        display_height
                    )
            else:
                imgui.text("No video frame available")
        
        imgui.end()
    
    def gui_emg(self):
        """GUI panel for EMG signal video."""
        if not self.show_emg or self.emg_player is None:
            return
        
        # Set window position and size
        imgui.set_next_window_position(
            self.emg_pos[0],
            self.emg_pos[1],
            imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_size(
            self.emg_size[0],
            self.emg_size[1],
            imgui.FIRST_USE_EVER
        )
        
        # Create window
        expanded, self.show_emg = imgui.begin(
            "EMG Signal##emg",
            self.show_emg
        )
        
        if expanded:
            # Get current frame from SMPLX sequence
            current_frame = self.scene.current_frame_id
            
            # Map frame to video frame
            video_frame_id = self._map_frame_to_video(
                current_frame,
                self.scene.n_frames,
                self.emg_player.total_frames
            )
            
            # Get video frame
            frame = self.emg_player.get_frame(video_frame_id)
            
            if frame is not None:
                # Update texture if frame changed
                if (self.emg_texture is None or 
                    self.emg_player.current_frame_id != video_frame_id):
                    self._update_video_texture(frame, "emg")
                    self.emg_player.current_frame_id = video_frame_id
                
                # Display frame info
                imgui.text(f"Frame: {video_frame_id}/{self.emg_player.total_frames - 1}")
                imgui.text(f"SMPLX Frame: {current_frame}/{self.scene.n_frames - 1}")
                
                # Display video frame
                if self.emg_texture is not None:
                    # Get available width
                    avail_width = imgui.get_content_region_available()[0]
                    
                    # Calculate display size maintaining aspect ratio
                    aspect_ratio = self.emg_player.width / max(1, self.emg_player.height)
                    display_height = min(avail_width / aspect_ratio, 250)
                    display_width = display_height * aspect_ratio
                    
                    # Display image using texture's OpenGL ID
                    # imgui.image expects the OpenGL texture ID
                    imgui.image(
                        self.emg_texture.glo,
                        display_width,
                        display_height
                    )
            else:
                imgui.text("No video frame available")
        
        imgui.end()
    
    def gui_cameras(self):
        """GUI panel for camera views."""
        if not self.cameras:
            return
        
        # Create window for each camera
        for i, camera in enumerate(self.cameras):
            if i not in self.show_cameras or not self.show_cameras[i]:
                continue
            
            # Set window position and size
            window_x = 50 + (i % 2) * 450
            window_y = 400 + (i // 2) * 350
            imgui.set_next_window_position(
                window_x,
                window_y,
                imgui.FIRST_USE_EVER
            )
            imgui.set_next_window_size(
                400,
                300,
                imgui.FIRST_USE_EVER
            )
            
            # Create window
            window_name = f"{camera.name}##camera_{i}"
            expanded, self.show_cameras[i] = imgui.begin(
                window_name,
                self.show_cameras[i]
            )
            
            if expanded:
                # Get current frame from SMPLX sequence
                current_frame = self.scene.current_frame_id
                
                # Load camera image
                frame = self._load_camera_image(i, current_frame)
                
                if frame is not None:
                    # Update texture if frame changed
                    if (i not in self.camera_textures or 
                        self.camera_textures[i] is None or
                        current_frame != getattr(camera, '_last_frame_id', -1)):
                        self._update_camera_texture(frame, i)
                        camera._last_frame_id = current_frame
                    
                    # Display frame info
                    imgui.text(f"Frame: {current_frame}/{self.scene.n_frames - 1}")
                    imgui.text(f"Camera: {camera.name}")
                    
                    # Display camera image
                    if i in self.camera_textures and self.camera_textures[i] is not None:
                        # Get available width
                        avail_width = imgui.get_content_region_available()[0]
                        
                        # Calculate display size maintaining aspect ratio
                        aspect_ratio = camera.cols / max(1, camera.rows)
                        display_height = min(avail_width / aspect_ratio, 250)
                        display_width = display_height * aspect_ratio
                        
                        # Display image using texture's OpenGL ID
                        imgui.image(
                            self.camera_textures[i].glo,
                            display_width,
                            display_height
                        )
                else:
                    imgui.text("No camera image available")
                    imgui.text(f"Camera images path: {self.camera_images_path}")
            
            imgui.end()
    
    def gui_camera_settings(self):
        """GUI panel for camera settings."""
        if not self.cameras:
            return
        
        # Set window position and size
        imgui.set_next_window_position(50, 50, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(300, 150, imgui.FIRST_USE_EVER)
        
        # Create window
        expanded, _ = imgui.begin("Camera Settings##camera_settings", True)
        
        if expanded:
            imgui.text(f"Total Cameras: {len(self.cameras)}")
            imgui.spacing()
            
            # Toggle frustum display for all cameras
            changed, self.show_camera_frustums = imgui.checkbox(
                "Show Camera Frustums", 
                self.show_camera_frustums
            )
            
            if changed:
                for i, camera in enumerate(self.cameras):
                    if self.show_camera_frustums:
                        try:
                            frustum_distance = camera.far if camera.far is not None else 10.0
                            camera.show_frustum(camera.cols, camera.rows, frustum_distance)
                        except Exception as e:
                            print(f"Warning: Could not show frustum for Camera {i}: {e}")
                    else:
                        try:
                            camera.hide_frustum()
                        except Exception as e:
                            print(f"Warning: Could not hide frustum for Camera {i}: {e}")
            
            imgui.spacing()
            imgui.text("Right-click on cameras in the")
            imgui.text("scene to toggle individual")
            imgui.text("camera frustums.")
        
        imgui.end()
    
    def _update_camera_texture(self, frame: np.ndarray, camera_index: int):
        """
        Update texture for camera image display.
        
        Args:
            frame: Camera image frame as numpy array (RGB format)
            camera_index: Index of the camera
        """
        if frame is None:
            return
        
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Ensure RGB format
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            height, width = frame.shape[:2]
            channels = 3
        else:
            return
        
        # Create or update texture
        try:
            if camera_index in self.camera_textures and self.camera_textures[camera_index] is not None:
                self.camera_textures[camera_index].release()
            
            self.camera_textures[camera_index] = self.ctx.texture(
                (width, height),
                channels,
                frame.tobytes()
            )
            # Register texture with imgui renderer (if method exists)
            if hasattr(self.imgui, 'register_texture'):
                self.imgui.register_texture(self.camera_textures[camera_index])
        except Exception as e:
            print(f"Error updating camera {camera_index} texture: {e}")
    
    def _map_frame_to_video(
        self,
        smplx_frame: int,
        smplx_total: int,
        video_total: int
    ) -> int:
        """
        Map SMPLX frame index to video frame index.
        
        Args:
            smplx_frame: Current SMPLX frame index
            smplx_total: Total SMPLX frames
            video_total: Total video frames
            
        Returns:
            Mapped video frame index
        """
        if video_total == 0 or smplx_total == 0:
            return 0
        
        # Simple linear mapping
        ratio = smplx_frame / max(1, smplx_total - 1)
        video_frame = int(ratio * (video_total - 1))
        return max(0, min(video_frame, video_total - 1))
    
    def _update_video_texture(self, frame: np.ndarray, video_type: str):
        """
        Update texture for video display.
        
        Args:
            frame: Video frame as numpy array (RGB format)
            video_type: "foot_pressure" or "emg"
        """
        if frame is None:
            return
        
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Ensure RGB format
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            height, width = frame.shape[:2]
            channels = 3
        else:
            return
        
        # Create or update texture
        try:
            if video_type == "foot_pressure":
                if self.foot_pressure_texture is not None:
                    self.foot_pressure_texture.release()
                self.foot_pressure_texture = self.ctx.texture(
                    (width, height),
                    channels,
                    frame.tobytes()
                )
                # Register texture with imgui renderer (if method exists)
                if hasattr(self.imgui, 'register_texture'):
                    self.imgui.register_texture(self.foot_pressure_texture)
            elif video_type == "emg":
                if self.emg_texture is not None:
                    self.emg_texture.release()
                self.emg_texture = self.ctx.texture(
                    (width, height),
                    channels,
                    frame.tobytes()
                )
                # Register texture with imgui renderer (if method exists)
                if hasattr(self.imgui, 'register_texture'):
                    self.imgui.register_texture(self.emg_texture)
        except Exception as e:
            print(f"Error updating {video_type} texture: {e}")
    
    def render(self, time, frame_time, export=False, transparent_background=False):
        """Override render to update video frames when frame changes."""
        # Call parent render method
        result = super().render(time, frame_time, export, transparent_background)
        
        # Update video frames if not exporting
        if not export:
            current_frame = self.scene.current_frame_id
            
            # Update foot pressure video
            if self.foot_pressure_player is not None:
                video_frame_id = self._map_frame_to_video(
                    current_frame,
                    self.scene.n_frames,
                    self.foot_pressure_player.total_frames
                )
                if self.foot_pressure_player.current_frame_id != video_frame_id:
                    frame = self.foot_pressure_player.get_frame(video_frame_id)
                    if frame is not None:
                        self._update_video_texture(frame, "foot_pressure")
                        self.foot_pressure_player.current_frame_id = video_frame_id
            
            # Update EMG video
            if self.emg_player is not None:
                video_frame_id = self._map_frame_to_video(
                    current_frame,
                    self.scene.n_frames,
                    self.emg_player.total_frames
                )
                if self.emg_player.current_frame_id != video_frame_id:
                    frame = self.emg_player.get_frame(video_frame_id)
                    if frame is not None:
                        self._update_video_texture(frame, "emg")
                        self.emg_player.current_frame_id = video_frame_id
            
            # Update camera images
            if self.cameras and self.camera_images_path:
                for i, camera in enumerate(self.cameras):
                    if i in self.show_cameras and self.show_cameras[i]:
                        last_frame = getattr(camera, '_last_frame_id', -1)
                        if current_frame != last_frame:
                            frame = self._load_camera_image(i, current_frame)
                            if frame is not None:
                                self._update_camera_texture(frame, i)
                                camera._last_frame_id = current_frame
        
        return result
    
    def release(self):
        """Clean up resources."""
        if self.foot_pressure_player is not None:
            self.foot_pressure_player.release()
        if self.emg_player is not None:
            self.emg_player.release()
        
        if self.foot_pressure_texture is not None:
            self.foot_pressure_texture.release()
        if self.emg_texture is not None:
            self.emg_texture.release()
        
        # Release camera textures
        for texture in self.camera_textures.values():
            if texture is not None:
                texture.release()
        
        super().release()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CAP Viewer - Review SMPLX with Foot Pressure & EMG")
    parser.add_argument(
        "smplx_path",
        type=str,
        help="Path to SMPLX animation file (.npz)"
    )
    parser.add_argument(
        "--foot-pressure",
        type=str,
        default=None,
        help="Path to foot pressure video file (optional)"
    )
    parser.add_argument(
        "--emg",
        type=str,
        default=None,
        help="Path to EMG signal video file (optional)"
    )
    
    args = parser.parse_args()
    
    # Create viewer
    viewer = CapViewer(
        smplx_path=args.smplx_path,
        foot_pressure_path=args.foot_pressure,
        emg_path=args.emg
    )
    
    viewer.run_animations = True
    viewer.playback_fps = 30.0
    
    try:
        viewer.run()
    finally:
        viewer.release()

