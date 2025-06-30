NVCC = nvcc
CC = g++-9

# OpenCV settings
OPENCV_INC = /usr/include/opencv4
OPENCV_LIB = -L/usr/lib/x86_64-linux-gnu
OPENCV_FLAGS := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_imgcodecs -lopencv_calib3d

# CUDA settings
CUDA_FLAGS = -std=c++17 -I$(OPENCV_INC) -ccbin $(CC)

# Source files
SRC = main.cu cuda1.cu cuda2.cu
OBJ = $(SRC:%.cu=build/%.o)

# Output binary
TARGET = bin/stitcher

# Default rule
all: $(TARGET)

# Build binary
$(TARGET): $(OBJ) | bin
	$(NVCC) $(CUDA_FLAGS) $(OBJ) $(OPENCV_LIB) $(OPENCV_FLAGS) -o $@

# Compile .cu to .o
build/%.o: %.cu | build
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

# Create directories if missing
build:
	mkdir -p build

bin:
	mkdir -p bin

# Clean build
clean:
	rm -rf build bin
