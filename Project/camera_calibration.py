
"""
========================================================================
CAMERA CALIBRATION PROJECT - Python OpenCV
Assignment 3 - CEN 575 / ECE 468 Computer Vision

Author: Eralba Spahija
Date: January 10, 2026

Implementation: Python with OpenCV Library
Algorithm: Zhang's Camera Calibration Method
========================================================================
"""

import cv2
import numpy as np
import os
import glob
from datetime import datetime

print("=" * 70)
print("         CAMERA CALIBRATION PROJECT - Python OpenCV")
print("=" * 70)
print()

# ======================== STEP 1: SETUP PATHS ========================
print("STEP 1: Setting up paths...")

# Define folder paths (relative to Project folder)
CHECKERBOARD_PATH = './DataSets/Checkerboards/'
IMAGES_PATH = './Results/unDistortedImages/'
RESULTS_PATH = './Results/'
UNDISTORTED_PATH = './DataSets/Images/'

# Create Results folders if they don't exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(UNDISTORTED_PATH, exist_ok=True)

print("  ✓ Paths configured successfully")
print()

# ==================== STEP 2: LOAD CALIBRATION IMAGES ====================
print("STEP 2: Loading calibration images...")

# Get list of checkerboard images
image_files = glob.glob(os.path.join(CHECKERBOARD_PATH, '*.jpg'))
image_files += glob.glob(os.path.join(CHECKERBOARD_PATH, '*.jpeg'))
image_files += glob.glob(os.path.join(CHECKERBOARD_PATH, '*.png'))

if not image_files:
    print("  ✗ ERROR: No images found in Checkerboards folder!")
    exit(1)

print(f"  ✓ Found {len(image_files)} checkerboard images")
print()

# ==================== STEP 3: DETECT CHECKERBOARD CORNERS ====================
print("STEP 3: Detecting checkerboard corners using OpenCV...")

# Checkerboard parameters
CHECKERBOARD = (7, 7)  # Internal corners (for 8x8 squares, use 7x7)
SQUARE_SIZE = 25.0     # Square size in mm 

print(f"  Pattern size: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} internal corners")
print(f"  Square size: {SQUARE_SIZE} mm")
print(f"  Using OpenCV function: cv2.findChessboardCorners()")
print()

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D coordinates in real world)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Storage for object points and image points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane
image_size = None
valid_images = []

# Process each image
for idx, fname in enumerate(image_files):
    img_name = os.path.basename(fname)
    print(f"  [{idx+1}/{len(image_files)}] Processing: {img_name}")
    
    # Load image
    img = cv2.imread(fname)
    if img is None:
        print(f"       ✗ Could not load image")
        continue
    
    # Store image size
    if image_size is None:
        image_size = (img.shape[1], img.shape[0])  # (width, height)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        # Refine corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Store object and image points
        objpoints.append(objp)
        imgpoints.append(corners2)
        valid_images.append(fname)
        
        print(f"       ✓ Checkerboard detected ({len(corners2)} corners)")
        
        # Draw and display the corners (VISUALIZATION)
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, CHECKERBOARD, corners2, ret)
        
        # Add text with image info
        cv2.putText(img_with_corners, f"Image {idx+1}/{len(image_files)}: {img_name}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_with_corners, f"Corners detected: {len(corners2)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the image with detected corners
        cv2.imshow('Checkerboard Detection', img_with_corners)
        cv2.waitKey(500)  # Show for 500ms then move to next image
        
    else:
        print(f"       ✗ Checkerboard not found")
        
        # Show failed detection
        img_failed = img.copy()
        cv2.putText(img_failed, f"Image {idx+1}/{len(image_files)}: FAILED", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_failed, "Checkerboard not detected", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Checkerboard Detection', img_failed)
        cv2.waitKey(500)

# Close the visualization window
cv2.destroyAllWindows()

num_valid = len(valid_images)
print()
print(f"  ✓ Successfully detected checkerboards in {num_valid}/{len(image_files)} images")
print()

if num_valid < 4:
    print(f"  ✗ ERROR: Need at least 4 valid images. Only found {num_valid}")
    cv2.destroyAllWindows()
    exit(1)

# Create a summary visualization of all detected checkerboards
print("  Creating calibration summary visualization...")
summary_images = []
max_summary_images = min(12, num_valid)  # Show up to 12 images

for i in range(max_summary_images):
    img = cv2.imread(valid_images[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        # Resize for summary grid
        img_small = cv2.resize(img, (400, 300))
        cv2.putText(img_small, f"Image {i+1}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        summary_images.append(img_small)

# Create grid layout
if len(summary_images) > 0:
    rows = (len(summary_images) + 3) // 4  # 4 images per row
    grid_rows = []
    
    for r in range(rows):
        start_idx = r * 4
        end_idx = min(start_idx + 4, len(summary_images))
        row_images = summary_images[start_idx:end_idx]
        
        # Pad with blank images if needed
        while len(row_images) < 4:
            blank = np.zeros((300, 400, 3), dtype=np.uint8)
            row_images.append(blank)
        
        grid_rows.append(np.hstack(row_images))
    
    # Stack all rows
    summary_grid = np.vstack(grid_rows)
    
    # Add title
    title_bar = np.zeros((60, summary_grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, f"Calibration Images: {num_valid} checkerboards detected", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    summary_grid = np.vstack([title_bar, summary_grid])
    
    # Display summary
    cv2.imshow('Calibration Summary', summary_grid)
    print("  ✓ Summary visualization displayed (press any key to continue)")
    cv2.waitKey(0)
    
    # Save summary
    cv2.imwrite(os.path.join(RESULTS_PATH, 'calibration_summary.png'), summary_grid)
    print("  ✓ Saved: calibration_summary.png")

cv2.destroyAllWindows()
print()

# ==================== STEP 4: CAMERA CALIBRATION ====================
print("STEP 4: Calibrating camera using OpenCV...")
print("  Using OpenCV function: cv2.calibrateCamera()")
print("  Algorithm: Zhang's Camera Calibration Method")
print()

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

# Calculate reprojection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                       camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)

print("  ✓ Calibration complete!")
print()

# ==================== STEP 5: DISPLAY RESULTS ====================
print("STEP 5: Displaying calibration results...")
print()

# Extract parameters
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]
k1, k2, p1, p2, k3 = dist_coeffs.ravel()

# Display results
print("=" * 60)
print("CAMERA CALIBRATION PARAMETERS (OpenCV)")
print("=" * 60)
print(f"Focal Length (fx, fy):     [{fx:.2f}, {fy:.2f}] pixels")
print(f"Principal Point (cx, cy):  [{cx:.2f}, {cy:.2f}] pixels")
print(f"Radial Distortion (k1):    {k1:.8f}")
print(f"Radial Distortion (k2):    {k2:.8f}")
print(f"Tangential Distortion (p1): {p1:.8f}")
print(f"Tangential Distortion (p2): {p2:.8f}")
print(f"Mean Reprojection Error:   {mean_error:.4f} pixels")
print("=" * 60)
print()

# ==================== STEP 6: SAVE RESULTS ====================
print("STEP 6: Saving calibration results...")

# Save to NumPy file
np.savez(os.path.join(RESULTS_PATH, 'camera_params.npz'),
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs,
         rvecs=rvecs,
         tvecs=tvecs,
         image_size=image_size,
         mean_error=mean_error)
print("  ✓ Saved: camera_params.npz")

# Save to text file
with open(os.path.join(RESULTS_PATH, 'Calibration Results.txt'), 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("CAMERA CALIBRATION RESULTS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Implementation: Python with OpenCV Library\n")
    f.write("Algorithm: Zhang's Camera Calibration Method\n")
    f.write("OpenCV Functions Used:\n")
    f.write("  - cv2.findChessboardCorners()\n")
    f.write("  - cv2.cornerSubPix()\n")
    f.write("  - cv2.calibrateCamera()\n")
    f.write("  - cv2.undistort()\n\n")
    f.write(f"Number of images used: {num_valid}\n")
    f.write(f"Checkerboard pattern: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} internal corners\n")
    f.write(f"Square size: {SQUARE_SIZE} mm\n\n")
    f.write("--- Intrinsic Parameters ---\n")
    f.write(f"Focal Length (fx, fy): [{fx:.4f}, {fy:.4f}] pixels\n")
    f.write(f"Principal Point (cx, cy): [{cx:.4f}, {cy:.4f}] pixels\n\n")
    f.write("--- Distortion Coefficients ---\n")
    f.write(f"Radial Distortion (k1, k2): [{k1:.8f}, {k2:.8f}]\n")
    f.write(f"Tangential Distortion (p1, p2): [{p1:.8f}, {p2:.8f}]\n\n")
    f.write("--- Calibration Quality ---\n")
    f.write(f"Mean Reprojection Error: {mean_error:.4f} pixels\n\n")
    f.write("--- Camera Matrix (Intrinsic Matrix) ---\n")
    f.write("K = \n")
    f.write(f"[{camera_matrix[0,0]:.4f}, {camera_matrix[0,1]:.4f}, {camera_matrix[0,2]:.4f}]\n")
    f.write(f"[{camera_matrix[1,0]:.4f}, {camera_matrix[1,1]:.4f}, {camera_matrix[1,2]:.4f}]\n")
    f.write(f"[{camera_matrix[2,0]:.4f}, {camera_matrix[2,1]:.4f}, {camera_matrix[2,2]:.4f}]\n\n")
    f.write("--- Distortion Coefficients Vector ---\n")
    f.write("dist_coeffs = [k1, k2, p1, p2, k3]\n")
    f.write(f"            = [{k1:.8f}, {k2:.8f}, {p1:.8f}, {p2:.8f}, {k3:.8f}]\n\n")
    f.write("=" * 70 + "\n")

print("  ✓ Saved: Calibration Results.txt")
print()

# ==================== STEP 7: UNDISTORT TEST IMAGES ====================
print("STEP 7: Undistorting test images using OpenCV...")
print("  Using OpenCV function: cv2.undistort()")
print()

# Get list of test images
test_images = glob.glob(os.path.join(IMAGES_PATH, '*.jpeg'))
test_images += glob.glob(os.path.join(IMAGES_PATH, '*.jpg'))
test_images += glob.glob(os.path.join(IMAGES_PATH, '*.png'))

if not test_images:
    print("  ⚠ No images found in Images folder to undistort")
    print()
else:
    print(f"  Found {len(test_images)} images to undistort")
    print()
    
    # Process each test image
    for idx, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)
        print(f"  [{idx+1}/{len(test_images)}] Processing: {img_name}")
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"       ✗ Could not load image")
                continue
            
            # Resize if needed
            h, w = img.shape[:2]
            if (w, h) != image_size:
                print(f"       Resizing: {w}x{h} → {image_size[0]}x{image_size[1]}")
                img = cv2.resize(img, image_size)
            
            # Undistort image
            undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
            
            # Calculate difference
            gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_undist = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            diff = np.abs(gray_orig.astype(float) - gray_undist.astype(float))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"       Correction: Max={max_diff:.1f} px, Mean={mean_diff:.2f} px")
            
            # Save undistorted image
            name, ext = os.path.splitext(img_name)
            output_name = f"{name}{ext}"
            output_path = os.path.join(UNDISTORTED_PATH, output_name)
            cv2.imwrite(output_path, undistorted)
            
            # Create comparison image
            # Add text to images
            img_labeled = img.copy()
            undist_labeled = undistorted.copy()
            
            cv2.putText(undist_labeled, f"Original (Max: {max_diff:.1f}px)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img_labeled, f"Undistorted (k1={k1:.4f})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            
            # Concatenate side by side
            comparison = np.hstack((undist_labeled, img_labeled))
            
            # Save comparison
            comparison_name = f"comparison_{name}.png"
            comparison_path = os.path.join(RESULTS_PATH, comparison_name)
            cv2.imwrite(comparison_path, comparison)
            
            print(f"       ✓ Saved undistorted image and comparison")
            print()
            
        except Exception as e:
            print(f"       ✗ ERROR: {str(e)}")
            print()
            continue
    
    print("  ✓ All images processed")
    print()

# ==================== COMPLETION ====================
print("=" * 70)
print("              ✓ PROJECT COMPLETE - Python OpenCV!")
print("=" * 70)
print()
print(f"Results saved to: {RESULTS_PATH}")
print(f"Undistorted images saved to: {UNDISTORTED_PATH}")
print()

print(f"Mean Reprojection Error: {mean_error:.4f} pixels ", end="")
if mean_error < 0.5:
    print("(EXCELLENT!)")
elif mean_error < 1.0:
    print("(GOOD)")
else:
    print("(Consider recalibration)")
print()
print("=" * 70)