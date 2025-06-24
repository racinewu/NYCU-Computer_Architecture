NVCC = nvcc
CC = g++-9

# OpenCV settings
OPENCV_INC = /usr/include/opencv4
OPENCV_LIB = -L/usr/lib/x86_64-linux-gnu
OPENCV_FLAGS := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_imgcodecs -lopencv_calib3d

# CUDA settings
CUDA_FLAGS = -std=c++17 -I$(OPENCV_INC) -ccbin $(CC)

SRC = main.cu cuda1.cu cuda2.cu

TARGET = stitcher

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CUDA_FLAGS) $(SRC) $(OPENCV_LIB) $(OPENCV_FLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)
