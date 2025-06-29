#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#define RANSAC_THRESHOLD 5.0f

using namespace std;
using namespace cv;

// 包含CUDA函數聲明
extern "C"
{
    void cuda_feature_matching(float *desc_left, float *desc_right, int desc_left_count, int desc_right_count, int desc_dim, float threshold, int *matches_left, int *matches_right, int *match_count);

    void cuda_ransac_homography(float *src_points, float *dst_points, int num_points, float *best_homography, int *max_inliers);

    void cuda_image_warping(unsigned char *img_right, unsigned char *stitch_img, int img_height, int img_width, int stitch_height, int stitch_width, float *inv_homography);

    void cuda_linear_blending(unsigned char *img_left, unsigned char *img_right, unsigned char *result, int height, int width, int left_width);
}

class CudaStitcher
{
private:
    Ptr<SIFT> sift_detector;

public:
    CudaStitcher()
    {
        sift_detector = SIFT::create();
    }

    ~CudaStitcher()
    {
        // CUDA cleanup handled in individual functions
    }

    // 主拼接函數
    Mat stitch(const string &left_path, const string &right_path,
                   const string &blending_mode, float threshold)
    {

        // 讀取圖像
        Mat img_left = imread(left_path);
        Mat img_right = imread(right_path);
        Mat gray_left, gray_right;

        cvtColor(img_left, gray_left, COLOR_BGR2GRAY);
        cvtColor(img_right, gray_right, COLOR_BGR2GRAY);

        cout << "Images loaded successfully..." << endl;

        // SIFT特征檢測
        vector<KeyPoint> kp_left, kp_right;
        Mat desc_left, desc_right;

        sift_detector->detectAndCompute(gray_left, noArray(), kp_left, desc_left);
        sift_detector->detectAndCompute(gray_right, noArray(), kp_right, desc_right);

        cout << "Left keypoints: " << kp_left.size() << endl;
        cout << "Right keypoints: " << kp_right.size() << endl;

        // CUDA特征匹配
        vector<pair<int, int>> matches = performCudaMatching(desc_left, desc_right, threshold);

        cout << "Number of matches: " << matches.size() << endl;

        // 繪製匹配結果
        drawMatchingPoint(img_left, img_right, kp_left, kp_right, matches);

        // 準備點對數據
        vector<Point2f> src_points, dst_points;
        for (const auto &match : matches)
        {
            src_points.push_back(kp_right[match.second].pt);
            dst_points.push_back(kp_left[match.first].pt);
        }

        // CUDA RANSAC求解單應性矩陣
        Mat homography = performCudaRANSAC(src_points, dst_points);

        // CUDA圖像變形和融合
        Mat result = performCudaWarping(img_left, img_right, homography, blending_mode);

        return result;
    }

private:
    // CUDA特征匹配
    vector<pair<int, int>> performCudaMatching(
        const Mat &desc_left, const Mat &desc_right, float threshold)
    {

        // 轉換描述符到float格式
        Mat desc_left_f, desc_right_f;
        desc_left.convertTo(desc_left_f, CV_32F);
        desc_right.convertTo(desc_right_f, CV_32F);

        int desc_left_count = desc_left_f.rows;
        int desc_right_count = desc_right_f.rows;
        int desc_dim = desc_left_f.cols;

        // 分配GPU內存並調用CUDA函數
        int *matches_left = new int[desc_left_count];
        int *matches_right = new int[desc_left_count];
        int match_count = 0;

        cuda_feature_matching(
            (float *)desc_left_f.data, (float *)desc_right_f.data,
            desc_left_count, desc_right_count, desc_dim, threshold,
            matches_left, matches_right, &match_count);

        // 整理匹配結果
        vector<pair<int, int>> matches;
        for (int i = 0; i < match_count; i++)
        {
            matches.push_back({matches_left[i], matches_right[i]});
        }

        delete[] matches_left;
        delete[] matches_right;

        return matches;
    }

    // CUDA RANSAC
    Mat performCudaRANSAC(const vector<Point2f> &src_points, const vector<Point2f> &dst_points)
    {

        int num_points = src_points.size();
        float *src_data = new float[num_points * 2];
        float *dst_data = new float[num_points * 2];

        // 準備數據
        for (int i = 0; i < num_points; i++)
        {
            src_data[i * 2] = src_points[i].x;
            src_data[i * 2 + 1] = src_points[i].y;
            dst_data[i * 2] = dst_points[i].x;
            dst_data[i * 2 + 1] = dst_points[i].y;
        }

        Mat H_opencv = findHomography(src_points, dst_points, RANSAC, RANSAC_THRESHOLD);
        cout << "===== opencv matrix ===== \n" << H_opencv << endl;

    float best_homography[9];  // 3x3 homography matrix (行優先存儲)
    int max_inliers;
    
    // 調用 CUDA RANSAC
    cuda_ransac_homography(src_data, dst_data, num_points, best_homography, &max_inliers);
    
    // 將結果轉換為 OpenCV Mat 格式
    Mat H = Mat(3, 3, CV_32F, best_homography).clone();
    

        delete[] src_data;
        delete[] dst_data;

        return H;
    }

    // CUDA圖像變形和融合
    Mat performCudaWarping(const Mat &img_left, const Mat &img_right, const Mat &homography, const string &blending_mode)
    {

        int hl = img_left.rows, wl = img_left.cols;
        int hr = img_right.rows, wr = img_right.cols;
        int stitch_height = max(hl, hr);
        int stitch_width = wl + wr;

        // 創建拼接結果圖像
        Mat stitch_img = Mat::zeros(stitch_height, stitch_width, CV_8UC3);

        // 計算逆單應性矩陣
        Mat inv_homography = homography.inv();
        inv_homography.convertTo(inv_homography, CV_32F);

        // 調用CUDA變形函數
        cuda_image_warping(
            img_right.data, stitch_img.data,
            hr, wr, stitch_height, stitch_width,
            (float *)inv_homography.data);

        cuda_linear_blending(img_left.data, stitch_img.data, stitch_img.data, stitch_height, stitch_width, wl);

        // 移除黑邊
        return removeBlackBorder(stitch_img);
    }

    // 繪製匹配結果
    void drawMatchingPoint(const Mat &img_left, const Mat &img_right,
                           const vector<KeyPoint> &kp_left,
                           const vector<KeyPoint> &kp_right,
                           const vector<pair<int, int>> &matches)
    {

        Mat vis;
        hconcat(img_left, img_right, vis);

        for (const auto &match : matches)
        {
            Point2f pt1 = kp_left[match.first].pt;
            Point2f pt2 = kp_right[match.second].pt;
            pt2.x += img_left.cols;

            circle(vis, pt1, 3, Scalar(0, 0, 255), 1);
            circle(vis, pt2, 3, Scalar(0, 255, 0), 1);
            line(vis, pt1, pt2, Scalar(255, 0, 0), 1);
        }

        imwrite("matching.jpg", vis);
        cout << "Matching visualization saved to matching.jpg" << endl;
    }

    // 移除黑邊
    Mat removeBlackBorder(const Mat &img)
    {
        int h = img.rows, w = img.cols;
        int reduced_h = h, reduced_w = w;

        // 從右到左
        for (int col = w - 1; col >= 0; col--)
        {
            bool all_black = true;
            for (int row = 0; row < h; row++)
            {
                Vec3b pixel = img.at<Vec3b>(row, col);
                if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
                {
                    all_black = false;
                    break;
                }
            }
            if (all_black)
                reduced_w--;
            else
                break;
        }

        // 從下到上
        for (int row = h - 1; row >= 0; row--)
        {
            bool all_black = true;
            for (int col = 0; col < reduced_w; col++)
            {
                Vec3b pixel = img.at<Vec3b>(row, col);
                if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
                {
                    all_black = false;
                    break;
                }
            }
            if (all_black)
                reduced_h--;
            else
                break;
        }

        return img(Rect(0, 0, reduced_w, reduced_h));
    }
};

int main(int argc, char *argv[])
{
    try
    {
        if (argc < 4)
        {
            cerr << "Usage: ./stitcher <input1.jpg> <input2.jpg> <output.jpg>\n";
            return 1;
        }

        // 檢查CUDA設備
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)
        {
            cerr << "No CUDA devices found!" << endl;
            return -1;
        }

        cout << "Found " << deviceCount << " CUDA device(s)" << endl;

        // 設置圖像路徑
        string left_path = argv[1];
        string right_path = argv[2];
        string stitch = argv[3];

        // 顯示原始圖像
        Mat img_left = imread(left_path);
        Mat img_right = imread(right_path);

        if (img_left.empty() || img_right.empty())
        {
            cerr << "Could not load images!" << endl;
            return -1;
        }

        //Mat total;
        //hconcat(img_left, img_right, total);
        // imshow("Total", total);

        // 執行CUDA拼接
        CudaStitcher stitcher;
        string blending_mode = "linearBlendingWithConstant";
        Mat result = stitcher.stitch(left_path, right_path, blending_mode, 0.75f);

        // 顯示和保存結果
        // imshow("stitch", result);
        imwrite(stitch, result);
        waitKey(0);

        cout << "Stitching completed successfully" << endl;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}