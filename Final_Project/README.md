# CUDA Accelerated Image Processing - Image Stitching
This project implements a CUDA-accelerated image stitching pipeline that leverages GPU parallelism for real-time performance. By offloading key stages-such as SIFT feature matching, RANSAC homography estimation, image warping, and blending-to the GPU, the program significantly reduces computation time compared to traditional CPU-only implementations. It is designed for high-resolution image mosaicking and optimized to run on modern NVIDIA GPUs.

## Problem Formulation
Given a set of high-resolution input images with overlapping visual content, the objective is to construct a seamless panoramic mosaic efficiently. The problem involves detecting and matching robust feature points across images, estimating geometric transformations to align them into a common coordinate system, and generating a visually consistent stitched output. Due to the high computational cost of feature extraction, matching, and geometric estimation, a GPU-accelerated solution is required. The goal is to leverage CUDA parallelism to significantly reduce execution time while preserving stitching accuracy and visual quality, enabling scalable and near real-time image mosaicking on modern NVIDIA GPUs.

## Features
- SIFT feature extraction and visualization of matches using Lowe's ratio test
- GPU-accelerated RANSAC-based homography estimation, warping, and constant-width linear blending on CUDA
- Automatic black border removal and image cropping
- OpenCV-based image I/O
- Modular architecture: separate build and bin folders

## Processing Pipeline
1.	**Image Loading**: Load two input images and convert them to grayscale and color formats.
2.	**Feature Extraction**: Detect SIFT keypoints and compute descriptors for both images.
3.	**Feature Matching**: Match descriptors using KNN search, then apply Lowe’s ratio test to retain valid matches.
4.	**Homography Estimation (GPU-accelerated)**: Use RANSAC to find the best homography matrix. Each iteration samples 4 matches ->Solve DLT -> counts inliers -> keeps the best model.
5.	**Image Warping (GPU-accelerated)**: Warp the right image using the estimated homography to align with the left image.
6.	**Blending (GPU-accelerated)**: Blend the aligned images with constant width
7.	**Post-processing**: Remove black borders and crop the final stitched image.

## Parameters

### Feature Matching
- **Lowe's ratio**: 0.75 (default threshold for ratio test)
- **SIFT detector**: OpenCV implementation with default parameters

### RANSAC Configuration
- **Threshold**: 1.0 pixels (distance for inlier classification)
- **Iterations**: 2000 (maximum RANSAC iterations)
- **Minimum matches**: 4 (required for homography estimation)
- **GPU threads**: Grid: 1D, Block Size: 256

### Warping
- **GPU threads**: Grid: 2D, Block Size: (16, 16)

### Alpha Mask
- **GPU threads**: Grid: 1D(Y-axis) Block Size: (1, 16)

### Blending Parameters
- **Constant width**: 10 pixels
- **Alpha interpolation**: Linear transition in overlap regions
- **GPU threads**: Grid: 2D, Block Size: (16, 16)

## Input / Output Format
### Input
- **Formats**: JPG, PNG
- **Requirements**: Two overlapping images with sufficient texture
- **Recommendations**: Images should have 20-50% overlap for best results

### Output
- Keypoints & Inliers: Prints `kp` and `matches` to the terminal.
- Feature Matches: Saves matched keypoints between the two input images to `Feature_Matches.jpg`
- Homography matrices: Prints the estimated homography matrix and its inverse to the terminal.
- Final Stitching Result: Saves Final panorama with specified blending mode applied to `output.jpg`.

## Environment
- OS: Ubuntu 22.04
- CUDA: nvcc 11.5
- Compiler: gcc 9.5
- OpenCV: 4.5.4
- C++ Standard: C++17

## Directory Structure
```
Final_Project/
  ├── Makefile  // Build script to compile the project
  ├── CA_Final/ // CUDA Tutorial
  ├── baseline/ // Original source images, m1~m6.jpg
  │
  ├── main.cu
  ├── cuda1.cu  // RANSAC and matching
  ├── cuda2.cu  // Warping and blending
  │
  ├── build/    // Object (.o) and dependency (.d) files created during build
  ├── bin/      // Final executable, e.g., bin/stitcher
  │
  └── README.md
```

## Usage Guide
### How to compile
To generate the executable `bin/stitcher`, simply run
```
make
```
### How to execute
Run the program with
```
./bin/stitcher <left_img> <right_img> <output_img>
```

## Experiment
<p align="center">
  <img src="baseline/feature_matches.jpg" alt="Feature Matches" width="800">
</p>
<p align="center">Figure 1. Feature Matches of m1 & m2</p>

<p align="center">
  <img src="baseline/output.jpg" alt="Stitched Result" width="800">
</p>
<p align="center">Figure 2. Stitched Result of m1 & m2</p>