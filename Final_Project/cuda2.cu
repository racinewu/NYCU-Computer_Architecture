#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdio.h>

#define THREADS_PER_BLOCK_2D 16
#define CONSTANT_BLEND_WIDTH 5
#define constant_width 10

// CUDA核函數：圖像變形
__global__ void warp_image_kernel(unsigned char *img_right, unsigned char *stitch_img, int img_height, int img_width, int stitch_height, int stitch_width, float *H)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= stitch_height || j >= stitch_width)
        return;

    // Coordinate (j, i, 1)
    float x = j, y = i, z = 1.0f;

    // Apply inverse homography: result = H * [x y 1]^T
    float denom = H[6] * x + H[7] * y + H[8];
    if (fabsf(denom) < 1e-8f)
        return;

    float src_x = (H[0] * x + H[1] * y + H[2]) / denom;
    float src_y = (H[3] * x + H[4] * y + H[5]) / denom;

    int rx = roundf(src_x);
    int ry = roundf(src_y);

    if (rx >= 0 && rx < img_width && ry >= 0 && ry < img_height)
    {
        int right_idx = (ry * img_width + rx) * 3;
        int stitch_idx = (i * stitch_width + j) * 3;

        stitch_img[stitch_idx + 0] = img_right[right_idx + 0];
        stitch_img[stitch_idx + 1] = img_right[right_idx + 1];
        stitch_img[stitch_idx + 2] = img_right[right_idx + 2];
    }
}

// CUDA核函數：線性融合
__global__ void linear_blending_kernel(
    unsigned char* img_left, unsigned char* img_right,
    unsigned char* result, float* alpha_mask,
    int height, int width, int wl)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= height || j >= width) return;
    if (j >= wl) return;  // 超出左圖範圍就跳過

    int result_idx = (i * width + j) * 3;
    int left_idx = (i * wl + j) * 3;
    float alpha = alpha_mask[i * width + j];

    if (alpha == 1.0f) {
        for (int c = 0; c < 3; c++) {
            result[result_idx + c] = img_left[left_idx + c];
        }
    }
    else if (alpha > 0.0f) {
        float beta = 1.0f - alpha;
        for (int c = 0; c < 3; c++) {
            result[result_idx + c] = (unsigned char)(
                alpha * img_left[left_idx + c] + beta * img_right[result_idx + c]
            );
        }
    }
    // alpha == 0: 不用動，右圖內容已在 result 初始化
}

// CUDA核函數：計算alpha掩碼（常數寬度線性融合）
// CUDA kernel to compute alpha mask for linear blending with constant width
__global__ void compute_alpha_mask_kernel(unsigned char *img_left, unsigned char *img_right, float *alpha_mask, int height, int width, int wl)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= height)
        return;

    // Find overlap region for this row
    int minIdx = -1, maxIdx = -1;

    for (int j = 0; j < width; j++)
    {
        bool left_nonzero = false, right_nonzero = false;
        if (j < wl)
        {
            int idx = (i * wl + j) * 3;
            bool left_nonzero = (img_left[idx] > 0 || img_left[idx + 1] > 0 || img_left[idx + 2] > 0);

            if (left_nonzero)
            {
                alpha_mask[i * width + j] = 1.0f; // 這裡補全不重疊區域
            }
        }

        // Check if left image pixel is non-zero (assuming 3 channels BGR)
        if (j < wl && i < height)
        {
            int idx = (i * wl + j) * 3;
            left_nonzero = (img_left[idx] > 0 || img_left[idx + 1] > 0 || img_left[idx + 2] > 0);
        }

        // Check if right image pixel is non-zero
        int idx_right = (i * width + j) * 3;
        right_nonzero = (img_right[idx_right] > 0 || img_right[idx_right + 1] > 0 || img_right[idx_right + 2] > 0);

        // Check for overlap
        if (left_nonzero && right_nonzero)
        {
            if (minIdx == -1)
                minIdx = j;
            maxIdx = j;
        }
    }

    // Compute alpha values for this row
    if (minIdx != -1 && maxIdx != -1 && minIdx != maxIdx)
    {
        float decrease_step = 1.0f / (maxIdx - minIdx);
        int middleIdx = (maxIdx + minIdx) / 2;

        // Left side
        for (int j = minIdx; j <= middleIdx; j++)
        {
            int alpha_idx = i * width + j;
            if (j >= middleIdx - constant_width)
            {
                alpha_mask[alpha_idx] = 1.0f - (decrease_step * (j - minIdx));
            }
            else
            {
                alpha_mask[alpha_idx] = 1.0f;
            }
        }

        // Right side
        for (int j = middleIdx + 1; j <= maxIdx; j++)
        {
            int alpha_idx = i * width + j;
            if (j <= middleIdx + constant_width)
            {
                alpha_mask[alpha_idx] = 1.0f - (decrease_step * (j - minIdx));
            }
            else
            {
                alpha_mask[alpha_idx] = 0.0f;
            }
        }
    }
}

// 主機函數：CUDA圖像變形
extern "C" void cuda_image_warping(unsigned char *img_right, unsigned char *stitch_img, int img_height, int img_width, int stitch_height, int stitch_width, const float *inv_homography)
{

    printf("Begin warping...\n");

    unsigned char *d_img_right, *d_stitch_img;
    float *d_inv_homography;

    size_t img_size = img_height * img_width * 3 * sizeof(unsigned char);
    size_t stitch_size = stitch_height * stitch_width * 3 * sizeof(unsigned char);
    size_t homography_size = 9 * sizeof(float);

    cudaMalloc(&d_img_right, img_size);
    cudaMalloc(&d_stitch_img, stitch_size);
    cudaMalloc(&d_inv_homography, homography_size);

    cudaMemcpy(d_img_right, img_right, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_homography, inv_homography, homography_size, cudaMemcpyHostToDevice);
    cudaMemset(d_stitch_img, 0, stitch_size);

    dim3 blockSize(16, 16);
    dim3 gridSize((stitch_width + blockSize.x - 1) / blockSize.x, (stitch_height + blockSize.y - 1) / blockSize.y);

    warp_image_kernel<<<gridSize, blockSize>>>(d_img_right, d_stitch_img, img_height, img_width, stitch_height, stitch_width, d_inv_homography);

    cudaDeviceSynchronize();
    cudaMemcpy(stitch_img, d_stitch_img, stitch_size, cudaMemcpyDeviceToHost);

    cudaFree(d_img_right);
    cudaFree(d_stitch_img);
    cudaFree(d_inv_homography);
}

// 主機函數：CUDA線性融合
extern "C" void cuda_linear_blending(unsigned char *img_left_data, unsigned char *img_right_data, unsigned char *result_data, int height, int width, int wl)
{
    printf("Begin linear blending...\n");

    // Device memory pointers
    unsigned char *d_img_left, *d_img_right, *d_result;
    float *d_alpha_mask;

    // Calculate sizes
    size_t left_size = height * wl * 3 * sizeof(unsigned char);
    size_t right_size = height * width * 3 * sizeof(unsigned char);
    size_t alpha_size = height * width * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_img_left, left_size);
    cudaMalloc(&d_img_right, right_size);
    cudaMalloc(&d_result, right_size);
    cudaMalloc(&d_alpha_mask, alpha_size);

    // Copy data to device
    cudaMemcpy(d_img_left, img_left_data, left_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_right, img_right_data, right_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, img_right_data, right_size, cudaMemcpyHostToDevice); // Initialize result with right image

    // Initialize alpha mask to zero
    cudaMemset(d_alpha_mask, 0, alpha_size);

    // Launch alpha mask computation kernel
    dim3 alpha_block(1, 16);
    dim3 alpha_grid(1, (height + alpha_block.y - 1) / alpha_block.y);
    compute_alpha_mask_kernel<<<alpha_grid, alpha_block>>>(d_img_left, d_img_right, d_alpha_mask, height, width, wl);

    // Launch linear blending kernel
    dim3 blend_block(16, 16);
    dim3 blend_grid((width + blend_block.x - 1) / blend_block.x, (height + blend_block.y - 1) / blend_block.y);
    linear_blending_kernel<<<blend_grid, blend_block>>>(d_img_left, d_img_right, d_result, d_alpha_mask, height, width, wl);

    // Copy result back to host
    cudaMemcpy(result_data, d_result, right_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_img_left);
    cudaFree(d_img_right);
    cudaFree(d_result);
    cudaFree(d_alpha_mask);

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}
