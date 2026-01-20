# CAP Viewer

CAP Viewer is a custom viewer built on top of aitviewer for reviewing SMPLX animations with synchronized foot pressure and EMG signal videos.

## Features

- **3D SMPLX Animation**: Main view displays SMPLX 3D model sequence
- **Foot Pressure Video**: Floating panel showing synchronized foot pressure video
- **EMG Signal Video**: Floating panel showing synchronized EMG signal video
- **Frame Synchronization**: All views are automatically synchronized to the same frame index
- **Interactive Controls**: 
  - Drag panels to reposition them
  - Use Playback panel to control animation
  - Video frames update automatically when sliding the progress bar

## Layout

The viewer uses a **floating panel layout** (方案3):
- Main window: Full 3D SMPLX animation view
- Floating panels: Foot Pressure and EMG videos (can be dragged and resized)

## Usage

### Basic Usage

```python
from capviewer import CapViewer

viewer = CapViewer(
    smplx_path="path/to/smplx_animation.npz",
    foot_pressure_path="path/to/foot_pressure.mp4",  # Optional
    emg_path="path/to/emg_signal.mp4"  # Optional
)

viewer.run_animations = True
viewer.playback_fps = 30.0
viewer.run()
```

### Command Line Usage

```bash
# With SMPLX only
python capviewer/capviewer.py path/to/smplx_animation.npz

# With SMPLX and foot pressure video
python capviewer/capviewer.py path/to/smplx_animation.npz --foot-pressure path/to/foot_pressure.mp4

# With SMPLX, foot pressure, and EMG videos
python capviewer/capviewer.py path/to/smplx_animation.npz --foot-pressure path/to/foot_pressure.mp4 --emg path/to/emg_signal.mp4
```

### Test Script

```bash
# Run test script (will try to find .npz files automatically)
python capviewer/test_capviewer.py
```

## Requirements

- aitviewer
- opencv-python (cv2)
- numpy
- imgui (included with aitviewer)

## Frame Synchronization

The viewer automatically maps SMPLX frame indices to video frame indices using linear interpolation:

- If SMPLX has N frames and video has M frames:
  - Frame 0 of SMPLX → Frame 0 of video
  - Frame N-1 of SMPLX → Frame M-1 of video
  - Intermediate frames are linearly interpolated

This ensures smooth synchronization even if the frame counts differ.

## Panel Controls

- **Reposition**: Click and drag the panel title bar
- **Resize**: Drag the edges/corners of the panel
- **Close/Show**: Click the X button or use the visibility toggle
- **Frame Info**: Each panel shows current frame numbers for both SMPLX and video

## Notes

- Video files are optional - the viewer will work with just SMPLX animation
- Video frames are cached for performance (up to 100 frames)
- The viewer automatically handles different frame counts between SMPLX and videos
- All panels are synchronized to the Playback control in the main viewer



