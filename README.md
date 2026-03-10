# Assignment 3: Camera Calibration

## Overview
Calibrate a phone camera using OpenCV's implementation of the Zhang algorithm.
The program detects chessboard corners across multiple images, computes the
intrinsic camera matrix and distortion coefficients, then uses them to undistort new photos.

## Folder Structure
```
ASSIGNMENT_3_[NAME_SURNAME]/
├── Project/
│   └── [source code files]
├── DataSets/
│   ├── Checkerboards/     # 15-20 calibration images
│   └── Images/            # photos to be undistorted
└── Results/
    ├── camera_params.txt  # intrinsic matrix + distortion coefficients
    └── Undistorted/       # output undistorted images
```

## Pipeline

1. **Image Acquisition** — Capture 15–20 chessboard photos from varied angles with good lighting
2. **Corner Detection** — Auto-detect and refine corners using `findChessboardCorners` + `cornerSubPix`
3. **Calibration** — Compute intrinsic matrix (fx, fy, cx, cy) and distortion coefficients (k1, k2)
4. **Undistortion** — Apply calibration results to new images

## Key OpenCV Functions

| Function | Purpose |
|---|---|
| `findChessboardCorners` | Detect inner corners of the chessboard |
| `cornerSubPix` | Refine corner positions to sub-pixel accuracy |
| `calibrateCamera` | Compute intrinsic matrix and distortion coefficients |
| `undistort` | Correct lens distortion on new images |

## Setup

**Python**
```bash
pip install opencv-python numpy
python calibrate.py
```


## Important
Make sure all paths in the code point to the correct folders (`DataSets/Checkerboards`, `DataSets/Images`, `Results/Undistorted`) so the code runs without modification.

## Resources
- [OpenCV Camera Calibration Docs](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Zhang Calibration Method](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/)

## Results - Camera Calibration
<img width="1600" height="960" alt="calibration_summary" src="https://github.com/user-attachments/assets/b71ad6c6-3456-47f7-ba57-14864f850d67" />
<img width="1920" height="1280" alt="comparison_test1" src="https://github.com/user-attachments/assets/e76067f5-2c2a-4692-b2ac-ee37503a72c0" />
