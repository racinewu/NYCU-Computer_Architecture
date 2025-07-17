# CUDA Accelerated Image Processing - Image Stitching

This program implements an image stitching pipeline using CUDA.

## Features
- GPU-accelerated RANSAC-based homography estimation, warping, and constant-width linear blending on CUDA
- Modular architecture: separate build and bin folders
- OpenCV-based image I/O and display

## Environment:
- OS: Ubuntu 22.04
- CUDA: nvcc 11.5
- Compiler: gcc 9.5
- OpenCV: 4.5.4
- C++ Standard: C++17

## Directory Structure
```
Final_Project/
  ├── main.cu        // Entry point of the CUDA-based image stitching pipeline
  ├── cuda1.cu       // GPU kernels for SIFT feature extraction and RANSAC homography estimation
  ├── cuda2.cu       // GPU kernels for image warping and blending
  ├── makefile       // Build script for compiling the CUDA project using NVCC
  ├── baseline/      // Original source images
  ├── build/         // Object files (.o), created after make
  └── bin/           // Final executable, e.g., bin/stitcher
```
## How to compile
To generate the executable `bin/stitcher`, simply run
```
make
```
## How to execute
Run the program with
```
./bin/stitcher <left_img.jpg> <right_img.jpg> <output.jpg>
```
