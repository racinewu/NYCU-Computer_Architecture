#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>
#include <stdio.h>

#define THREADS_PER_BLOCK 128
#define RANSAC_THRESHOLD 4.5f
#define PAIRS 4
#define RANSAC_ITER 2000
#define MIN_DISTANCE_SQUARED 5.0f  // 最小距離平方
#define COLLINEARITY_THRESHOLD 1.0f // 共線性閾值

// CUDA kernel for random number generation initialization
__global__ void init_curand_kernel(curandState *state, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// Device function to normalize points (improve numerical stability)
__device__ void normalize_points(float* points, int num_points, float* normalized_points, 
                                float* transform_matrix) {
    // Compute centroid
    float cx = 0.0f, cy = 0.0f;
    for (int i = 0; i < num_points; i++) {
        cx += points[i * 2];
        cy += points[i * 2 + 1];
    }
    cx /= num_points;
    cy /= num_points;
    
    // Compute RMS distance to centroid
    float rms_dist = 0.0f;
    for (int i = 0; i < num_points; i++) {
        float dx = points[i * 2] - cx;
        float dy = points[i * 2 + 1] - cy;
        rms_dist += dx * dx + dy * dy;
    }
    rms_dist = sqrtf(rms_dist / num_points);
    
    // Avoid division by zero
    if (rms_dist < 1e-10f) rms_dist = 1.0f;
    
    float scale = 1.414213562f / rms_dist;  // sqrt(2) / rms_dist
    
    // Normalize points
    for (int i = 0; i < num_points; i++) {
        normalized_points[i * 2] = (points[i * 2] - cx) * scale;
        normalized_points[i * 2 + 1] = (points[i * 2 + 1] - cy) * scale;
    }
    
    // Build transformation matrix
    transform_matrix[0] = scale; transform_matrix[1] = 0.0f;   transform_matrix[2] = -cx * scale;
    transform_matrix[3] = 0.0f;  transform_matrix[4] = scale; transform_matrix[5] = -cy * scale;
    transform_matrix[6] = 0.0f;  transform_matrix[7] = 0.0f;  transform_matrix[8] = 1.0f;
}

// Device function to multiply 3x3 matrices
__device__ void multiply_3x3(float* A, float* B, float* C) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
            }
        }
    }
}

// Device function to invert 3x3 matrix
__device__ bool invert_3x3(float* matrix, float* result) {
    float det = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
                matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
                matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    
    if (fabsf(det) < 1e-10f) return false;
    
    float inv_det = 1.0f / det;
    
    result[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inv_det;
    result[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inv_det;
    result[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inv_det;
    result[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inv_det;
    result[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inv_det;
    result[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inv_det;
    result[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inv_det;
    result[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inv_det;
    result[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inv_det;
    
    return true;
}

// 改進的點配置檢查函數
__device__ bool check_point_configuration(float* src_points, float* dst_points) {
    // 1. 檢查點之間的最小距離
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            float dx_src = src_points[i*2] - src_points[j*2];
            float dy_src = src_points[i*2+1] - src_points[j*2+1];
            float dx_dst = dst_points[i*2] - dst_points[j*2];
            float dy_dst = dst_points[i*2+1] - dst_points[j*2+1];
            
            if (dx_src*dx_src + dy_src*dy_src < MIN_DISTANCE_SQUARED ||
                dx_dst*dx_dst + dy_dst*dy_dst < MIN_DISTANCE_SQUARED) {
                return false;
            }
        }
    }
    
    // 2. 更嚴格的共線性檢查
    for (int set = 0; set < 2; set++) { // 檢查src和dst
        float* points = (set == 0) ? src_points : dst_points;
        
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                for (int k = j + 1; k < 4; k++) {
                    float x1 = points[i*2], y1 = points[i*2+1];
                    float x2 = points[j*2], y2 = points[j*2+1];
                    float x3 = points[k*2], y3 = points[k*2+1];
                    
                    // 計算面積（使用交叉乘積）
                    float area = fabsf((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1));
                    if (area < COLLINEARITY_THRESHOLD) {
                        return false;
                    }
                }
            }
        }
    }
    
    // 3. 檢查四邊形的凸性
    for (int set = 0; set < 2; set++) {
        float* points = (set == 0) ? src_points : dst_points;
        
        // 計算四個頂點的交叉乘積符號
        int sign_changes = 0;
        int prev_sign = 0;
        
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            int k = (i + 2) % 4;
            
            float x1 = points[i*2], y1 = points[i*2+1];
            float x2 = points[j*2], y2 = points[j*2+1];
            float x3 = points[k*2], y3 = points[k*2+1];
            
            float cross = (x2-x1)*(y3-y2) - (y2-y1)*(x3-x2);
            int curr_sign = (cross > 0) ? 1 : -1;
            
            if (i > 0 && curr_sign != prev_sign) {
                sign_changes++;
            }
            prev_sign = curr_sign;
        }
        
        // 凸四邊形應該沒有符號變化或只有兩次變化
        if (sign_changes > 2) {
            return false;
        }
    }
    
    return true;
}

// 改進的SVD分解求解homography
__device__ bool solve_homography_svd(float* src_points, float* dst_points, float* H) {
    if (!check_point_configuration(src_points, dst_points)) {
        return false;
    }
    
    // 正規化點
    float norm_src[8], norm_dst[8];
    float T1[9], T2[9];
    
    normalize_points(src_points, 4, norm_src, T1);
    normalize_points(dst_points, 4, norm_dst, T2);
    
    // 構建矩陣A
    float A[8][9];
    for (int r = 0; r < 4; r++) {
        float x = norm_src[r * 2];
        float y = norm_src[r * 2 + 1];
        float u = norm_dst[r * 2];
        float v = norm_dst[r * 2 + 1];
        
        // 第一行: [0, 0, 0, -x, -y, -1, vx, vy, v]
        A[r * 2][0] = 0.0f;     A[r * 2][1] = 0.0f;     A[r * 2][2] = 0.0f;
        A[r * 2][3] = -x;       A[r * 2][4] = -y;       A[r * 2][5] = -1.0f;
        A[r * 2][6] = v * x;    A[r * 2][7] = v * y;    A[r * 2][8] = v;
        
        // 第二行: [x, y, 1, 0, 0, 0, -ux, -uy, -u]
        A[r * 2 + 1][0] = x;      A[r * 2 + 1][1] = y;      A[r * 2 + 1][2] = 1.0f;
        A[r * 2 + 1][3] = 0.0f;   A[r * 2 + 1][4] = 0.0f;   A[r * 2 + 1][5] = 0.0f;
        A[r * 2 + 1][6] = -u * x; A[r * 2 + 1][7] = -u * y; A[r * 2 + 1][8] = -u;
    }
    
    // 計算A^T * A
    float ATA[9][9];
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            ATA[i][j] = 0.0f;
            for (int k = 0; k < 8; k++) {
                ATA[i][j] += A[k][i] * A[k][j];
            }
        }
    }
    
    // 使用改進的反冪次方法求最小特徵向量
    float h[9] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 1.0f};
    float prev_h[9];
    
    for (int iter = 0; iter < 100; iter++) {
        // 保存前一次結果
        for (int i = 0; i < 9; i++) prev_h[i] = h[i];
        
        float h_new[9] = {0.0f};
        float epsilon = 1e-6f;
        
        // 使用共軛梯度法求解 (ATA + εI)h_new = h
        for (int cg_iter = 0; cg_iter < 30; cg_iter++) {
            for (int i = 0; i < 9; i++) {
                float sum = h[i];
                for (int j = 0; j < 9; j++) {
                    if (i != j) {
                        sum -= ATA[i][j] * h_new[j];
                    }
                }
                h_new[i] = sum / (ATA[i][i] + epsilon);
            }
        }
        
        // 正規化
        float norm = 0.0f;
        for (int i = 0; i < 9; i++) {
            norm += h_new[i] * h_new[i];
        }
        norm = sqrtf(norm);
        
        if (norm < 1e-12f) return false;
        
        for (int i = 0; i < 9; i++) {
            h[i] = h_new[i] / norm;
        }
        
        // 檢查收斂性
        float diff = 0.0f;
        for (int i = 0; i < 9; i++) {
            float d = h[i] - prev_h[i];
            diff += d * d;
        }
        
        if (diff < 1e-10f) break;
    }
    
    // 重塑為3x3矩陣
    float H_norm[9];
    for (int i = 0; i < 9; i++) {
        H_norm[i] = h[i];
    }
    
    // 去正規化: H = T2^(-1) * H_norm * T1
    float T2_inv[9];
    if (!invert_3x3(T2, T2_inv)) {
        return false;
    }
    
    float temp[9];
    multiply_3x3(H_norm, T1, temp);
    multiply_3x3(T2_inv, temp, H);
    
    // 正規化使H[8] = 1
    if (fabsf(H[8]) > 1e-12f) {
        for (int i = 0; i < 9; i++) {
            H[i] /= H[8];
        }
    } else {
        return false;
    }
    
    return true;
}

// 改進的重投影誤差計算
__device__ float compute_reprojection_error(float* H, float* src_point, float* dst_point) {
    float x = src_point[0];
    float y = src_point[1];
    
    // 應用homography
    float u_prime = H[0] * x + H[1] * y + H[2];
    float v_prime = H[3] * x + H[4] * y + H[5];
    float w_prime = H[6] * x + H[7] * y + H[8];
    
    // 檢查有效的齊次坐標
    if (fabsf(w_prime) < 1e-12f) {
        return 1e6f;
    }
    
    // 轉換為笛卡爾坐標
    float u = u_prime / w_prime;
    float v = v_prime / w_prime;
    
    // 檢查結果的合理性
    if (!isfinite(u) || !isfinite(v)) {
        return 1e6f;
    }
    
    // 計算雙向誤差（對稱誤差）
    float dx = u - dst_point[0];
    float dy = v - dst_point[1];
    float forward_error = dx * dx + dy * dy;
    
    // 反向誤差（從dst到src）
    float H_inv[9];
    // 這裡簡化，只計算前向誤差
    // 實際應用中可以計算逆矩陣並計算雙向誤差
    
    return sqrtf(forward_error);
}

// 改進的採樣策略
__device__ void sample_points_improved(int num_points, int* sample_indices, 
                                     curandState* local_state, int tid) {
    // 使用分層採樣確保點的分佈性
    if (num_points >= 16) {
        // 將點空間分為4個象限，每象限採樣一個點
        int quadrant_size = num_points / 4;
        for (int i = 0; i < 4; i++) {
            int start = i * quadrant_size;
            int end = (i == 3) ? num_points : (i + 1) * quadrant_size;
            sample_indices[i] = start + (curand(local_state) % (end - start));
        }
    } else {
        // 標準隨機採樣
        for (int i = 0; i < PAIRS; i++) {
            bool valid = false;
            int attempts = 0;
            while (!valid && attempts < 50) {
                sample_indices[i] = curand(local_state) % num_points;
                valid = true;
                for (int j = 0; j < i; j++) {
                    if (sample_indices[i] == sample_indices[j]) {
                        valid = false;
                        break;
                    }
                }
                attempts++;
            }
            if (attempts >= 50) {
                sample_indices[i] = (tid * PAIRS + i) % num_points;
            }
        }
    }
}

// CUDA kernel for RANSAC homography estimation
__global__ void ransac_homography_kernel(float* src_points, float* dst_points, int num_points,
                                       float* all_H, int* all_inliers, float* all_scores, 
                                       curandState *states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= RANSAC_ITER) return;
    
    curandState local_state = states[tid];
    
    // 改進的採樣
    int sample_indices[PAIRS];
    sample_points_improved(num_points, sample_indices, &local_state, tid);
    
    // 提取採樣點
    float sampled_src[8], sampled_dst[8];
    for (int i = 0; i < PAIRS; i++) {
        int idx = sample_indices[i];
        sampled_src[i * 2] = src_points[idx * 2];
        sampled_src[i * 2 + 1] = src_points[idx * 2 + 1];
        sampled_dst[i * 2] = dst_points[idx * 2];
        sampled_dst[i * 2 + 1] = dst_points[idx * 2 + 1];
    }
    
    // 求解homography
    float local_H[9];
    bool success = solve_homography_svd(sampled_src, sampled_dst, local_H);
    
    int inlier_count = 0;
    float total_error = 0.0f;
    float quality_score = 0.0f;
    
    if (success) {
        // 驗證homography的數值穩定性
        bool valid = true;
        for (int i = 0; i < 9; i++) {
            if (!isfinite(local_H[i]) || fabsf(local_H[i]) > 1e6f) {
                valid = false;
                break;
            }
        }
        
        // 檢查條件數（簡化檢查）
        if (valid) {
            float det = local_H[0] * (local_H[4] * local_H[8] - local_H[5] * local_H[7]) -
                       local_H[1] * (local_H[3] * local_H[8] - local_H[5] * local_H[6]) +
                       local_H[2] * (local_H[3] * local_H[7] - local_H[4] * local_H[6]);
            
            if (fabsf(det) < 1e-10f) {
                valid = false;
            }
        }
        
        if (valid) {
            // 計算內點和質量分數
            for (int i = 0; i < num_points; i++) {
                float src_pt[2] = {src_points[i * 2], src_points[i * 2 + 1]};
                float dst_pt[2] = {dst_points[i * 2], dst_points[i * 2 + 1]};
                
                float error = compute_reprojection_error(local_H, src_pt, dst_pt);
                
                if (error < RANSAC_THRESHOLD) {
                    inlier_count++;
                    total_error += error;
                } else {
                    // 軟約束：給接近閾值的點一些權重
                    if (error < RANSAC_THRESHOLD * 2.0f) {
                        float weight = 1.0f - (error - RANSAC_THRESHOLD) / RANSAC_THRESHOLD;
                        quality_score += weight * 0.5f;
                    }
                }
            }
            
            // 計算質量分數
            if (inlier_count > 0) {
                quality_score += inlier_count;
                quality_score -= total_error / inlier_count; // 懲罰高誤差
            }
        }
    }
    
    // 存儲結果
    for (int i = 0; i < 9; i++) {
        all_H[tid * 9 + i] = success ? local_H[i] : 0.0f;
    }
    all_inliers[tid] = inlier_count;
    all_scores[tid] = quality_score;
    
    states[tid] = local_state;
}

// Host function to call CUDA RANSAC
extern "C" void cuda_ransac_homography(float* src_data, float* dst_data, int num_points, 
                                     float* best_homography, int* max_inliers) {
    if (num_points < 4) {
        printf("Error: Need at least 4 points for homography estimation\n");
        *max_inliers = 0;
        return;
    }
    
    // Device memory allocation
    float *d_src_points, *d_dst_points;
    float *d_all_H;
    int *d_all_inliers;
    float *d_all_scores;
    curandState *d_states;
    
    size_t points_size = num_points * 2 * sizeof(float);
    size_t all_H_size = RANSAC_ITER * 9 * sizeof(float);
    size_t all_inliers_size = RANSAC_ITER * sizeof(int);
    size_t all_scores_size = RANSAC_ITER * sizeof(float);
    
    cudaMalloc(&d_src_points, points_size);
    cudaMalloc(&d_dst_points, points_size);
    cudaMalloc(&d_all_H, all_H_size);
    cudaMalloc(&d_all_inliers, all_inliers_size);
    cudaMalloc(&d_all_scores, all_scores_size);
    cudaMalloc(&d_states, RANSAC_ITER * sizeof(curandState));
    
    // Copy data to device
    cudaMemcpy(d_src_points, src_data, points_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_points, dst_data, points_size, cudaMemcpyHostToDevice);
    
    // Initialize random number generator with better seed
    dim3 init_block(256);
    dim3 init_grid((RANSAC_ITER + init_block.x - 1) / init_block.x);
    unsigned long seed = time(NULL) + rand();
    init_curand_kernel<<<init_grid, init_block>>>(d_states, seed, RANSAC_ITER);
    
    cudaDeviceSynchronize();
    
    // Launch RANSAC kernel
    dim3 block(256);
    dim3 grid((RANSAC_ITER + block.x - 1) / block.x);
    ransac_homography_kernel<<<grid, block>>>(d_src_points, d_dst_points, num_points,
                                            d_all_H, d_all_inliers, d_all_scores, d_states);
    
    cudaDeviceSynchronize();
    
    // Copy results back to host
    float* host_all_H = new float[RANSAC_ITER * 9];
    int* host_all_inliers = new int[RANSAC_ITER];
    float* host_all_scores = new float[RANSAC_ITER];
    
    cudaMemcpy(host_all_H, d_all_H, all_H_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_all_inliers, d_all_inliers, all_inliers_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_all_scores, d_all_scores, all_scores_size, cudaMemcpyDeviceToHost);
    
    // 多重準則選擇最佳結果
    *max_inliers = 0;
    int best_idx = 0;
    float best_score = -1e6f;
    
    for (int i = 0; i < RANSAC_ITER; i++) {
        bool is_better = false;
        
        // 首先考慮內點數量
        if (host_all_inliers[i] > *max_inliers) {
            is_better = true;
        } 
        // 如果內點數量相近，考慮質量分數
        else if (host_all_inliers[i] >= *max_inliers * 0.95f && 
                 host_all_scores[i] > best_score) {
            is_better = true;
        }
        
        if (is_better) {
            *max_inliers = host_all_inliers[i];
            best_score = host_all_scores[i];
            best_idx = i;
        }
    }
    
    // Copy best homography
    for (int i = 0; i < 9; i++) {
        best_homography[i] = host_all_H[best_idx * 9 + i];
    }
    
    printf("CUDA RANSAC Maximum inliers: %d, Best score: %.3f\n", 
           *max_inliers, best_score);
    printf("===== Best homography =====\n");
    for (int i = 0; i < 3; i++) {
        printf("[%.6f, %.6f, %.6f]\n", 
               best_homography[i*3], best_homography[i*3+1], best_homography[i*3+2]);
    }
    
    // Clean up
    delete[] host_all_H;
    delete[] host_all_inliers;
    delete[] host_all_scores;
    
    cudaFree(d_src_points);
    cudaFree(d_dst_points);
    cudaFree(d_all_H);
    cudaFree(d_all_inliers);
    cudaFree(d_all_scores);
    cudaFree(d_states);
}

// CUDA核函數：計算描述符之間的距離
__global__ void compute_descriptor_distances(float *desc_left, float *desc_right, int desc_left_count, int desc_right_count, int desc_dim, float *distances)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= desc_left_count * desc_right_count)
        return;

    int left_idx = idx / desc_right_count;
    int right_idx = idx % desc_right_count;

    float dist = 0.0f;
    for (int d = 0; d < desc_dim; d++)
    {
        float diff = desc_left[left_idx * desc_dim + d] - desc_right[right_idx * desc_dim + d];
        dist += diff * diff;
    }

    distances[idx] = sqrtf(dist);
}

// CUDA核函數：尋找最佳匹配
__global__ void find_best_matches(float *distances, int desc_left_count, int desc_right_count, float threshold, int *matches_left, int *matches_right, int *match_count, int *valid_matches)
{

    int left_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (left_idx >= desc_left_count)
        return;

    float min_dist = FLT_MAX;
    float second_min_dist = FLT_MAX;
    int best_match = -1;

    // 找到最小和次小距離
    for (int right_idx = 0; right_idx < desc_right_count; right_idx++)
    {
        float dist = distances[left_idx * desc_right_count + right_idx];
        if (dist < min_dist)
        {
            second_min_dist = min_dist;
            min_dist = dist;
            best_match = right_idx;
        }
        else if (dist < second_min_dist)
        {
            second_min_dist = dist;
        }
    }

    // Lowe's ratio test
    valid_matches[left_idx] = 0;
    if (best_match != -1 && min_dist <= threshold * second_min_dist)
    {
        matches_left[left_idx] = left_idx;
        matches_right[left_idx] = best_match;
        valid_matches[left_idx] = 1;
    }
}


// 主機函數：CUDA特征匹配
extern "C" void cuda_feature_matching(float *desc_left, float *desc_right, int desc_left_count, int desc_right_count, int desc_dim, float threshold, int *matches_left, int *matches_right, int *match_count)
{

    // 分配GPU內存
    float *d_desc_left, *d_desc_right, *d_distances;
    int *d_matches_left, *d_matches_right, *d_valid_matches, *d_match_count;

    size_t desc_left_size = desc_left_count * desc_dim * sizeof(float);
    size_t desc_right_size = desc_right_count * desc_dim * sizeof(float);
    size_t distances_size = desc_left_count * desc_right_count * sizeof(float);
    size_t matches_size = desc_left_count * sizeof(int);

    cudaMalloc(&d_desc_left, desc_left_size);
    cudaMalloc(&d_desc_right, desc_right_size);
    cudaMalloc(&d_distances, distances_size);
    cudaMalloc(&d_matches_left, matches_size);
    cudaMalloc(&d_matches_right, matches_size);
    cudaMalloc(&d_valid_matches, matches_size);
    cudaMalloc(&d_match_count, sizeof(int));

    // 複製數據到GPU
    cudaMemcpy(d_desc_left, desc_left, desc_left_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_desc_right, desc_right, desc_right_size, cudaMemcpyHostToDevice);
    cudaMemset(d_match_count, 0, sizeof(int));

    // 計算距離
    int total_pairs = desc_left_count * desc_right_count;
    int blocks_distances = (total_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    compute_descriptor_distances<<<blocks_distances, THREADS_PER_BLOCK>>>(d_desc_left, d_desc_right, desc_left_count, desc_right_count, desc_dim, d_distances);

    // 找到最佳匹配
    int blocks_matches = (desc_left_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    find_best_matches<<<blocks_matches, THREADS_PER_BLOCK>>>(
        d_distances, desc_left_count, desc_right_count, threshold,
        d_matches_left, d_matches_right, d_match_count, d_valid_matches);

    // 複製結果回主機
    int *h_valid_matches = new int[desc_left_count];
    cudaMemcpy(matches_left, d_matches_left, matches_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(matches_right, d_matches_right, matches_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valid_matches, d_valid_matches, matches_size, cudaMemcpyDeviceToHost);

    // 計算有效匹配數量
    *match_count = 0;
    for (int i = 0; i < desc_left_count; i++)
    {
        if (h_valid_matches[i])
        {
            matches_left[*match_count] = matches_left[i];
            matches_right[*match_count] = matches_right[i];
            (*match_count)++;
        }
    }

    // 清理GPU內存
    cudaFree(d_desc_left);
    cudaFree(d_desc_right);
    cudaFree(d_distances);
    cudaFree(d_matches_left);
    cudaFree(d_matches_right);
    cudaFree(d_valid_matches);
    cudaFree(d_match_count);

    delete[] h_valid_matches;
}
