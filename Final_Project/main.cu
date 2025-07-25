#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#define RANSAC_THRESHOLD 5.0f

using namespace std;
using namespace cv;

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

    Mat stitch(const string &left_path, const string &right_path,
               const string &blending_mode, float threshold)
    {
        Mat img_left = imread(left_path);
        Mat img_right = imread(right_path);
        Mat gray_left, gray_right;

        cvtColor(img_left, gray_left, COLOR_BGR2GRAY);
        cvtColor(img_right, gray_right, COLOR_BGR2GRAY);

        cout << "Images loaded successfully..." << endl;

        // SIFT
        vector<KeyPoint> kp_left, kp_right;
        Mat desc_left, desc_right;

        sift_detector->detectAndCompute(gray_left, noArray(), kp_left, desc_left);
        sift_detector->detectAndCompute(gray_right, noArray(), kp_right, desc_right);

        cout << "Left keypoints: " << kp_left.size() << endl;
        cout << "Right keypoints: " << kp_right.size() << endl;

        vector<pair<int, int>> matches = performCudaMatching(desc_left, desc_right, threshold);

        cout << "Number of matches: " << matches.size() << endl;

        // Draw matches
        drawMatchingPoint(img_left, img_right, kp_left, kp_right, matches);

        vector<Point2f> src_points, dst_points;
        for (const auto &match : matches)
        {
            src_points.push_back(kp_right[match.second].pt);
            dst_points.push_back(kp_left[match.first].pt);
        }

        Mat homography = performCudaRANSAC(src_points, dst_points);

        Mat result = performCudaWarping(img_left, img_right, homography, blending_mode);

        return result;
    }

private:
    vector<pair<int, int>> performCudaMatching(
        const Mat &desc_left, const Mat &desc_right, float threshold)
    {

        Mat desc_left_f, desc_right_f;
        desc_left.convertTo(desc_left_f, CV_32F);
        desc_right.convertTo(desc_right_f, CV_32F);

        int desc_left_count = desc_left_f.rows;
        int desc_right_count = desc_right_f.rows;
        int desc_dim = desc_left_f.cols;

        int *matches_left = new int[desc_left_count];
        int *matches_right = new int[desc_left_count];
        int match_count = 0;

        cuda_feature_matching(
            (float *)desc_left_f.data, (float *)desc_right_f.data,
            desc_left_count, desc_right_count, desc_dim, threshold,
            matches_left, matches_right, &match_count);

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

        for (int i = 0; i < num_points; i++)
        {
            src_data[i * 2] = src_points[i].x;
            src_data[i * 2 + 1] = src_points[i].y;
            dst_data[i * 2] = dst_points[i].x;
            dst_data[i * 2 + 1] = dst_points[i].y;
        }

        Mat H_opencv = findHomography(src_points, dst_points, RANSAC, RANSAC_THRESHOLD);
        cout << " ====== Opencv matrix ====== " << endl;
        for (int i = 0; i < H_opencv.rows; ++i)
        {
            for (int j = 0; j < H_opencv.cols; ++j)
            {
                cout << fixed << setw(8) << setprecision(3) << H_opencv.at<double>(i, j) << " ";
            }
            cout << endl;
        }

        float best_homography[9];
        int max_inliers;

        cuda_ransac_homography(src_data, dst_data, num_points, best_homography, &max_inliers);

        // Trsform to OpenCV Mat form
        Mat H = Mat(3, 3, CV_32F, best_homography).clone();
        cout << " ===== Best homography ===== " << endl;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                cout << fixed << setw(8) << setprecision(3)
                     << H.at<float>(i, j) << " ";
            }
            cout << endl;
        }

        delete[] src_data;
        delete[] dst_data;

        return H;
    }

    Mat performCudaWarping(const Mat &img_left, const Mat &img_right, const Mat &homography, const string &blending_mode)
    {

        int hl = img_left.rows, wl = img_left.cols;
        int hr = img_right.rows, wr = img_right.cols;
        int stitch_height = max(hl, hr);
        int stitch_width = wl + wr;

        // Create stich image
        Mat stitch_img = Mat::zeros(stitch_height, stitch_width, CV_8UC3);

        // Compute inverse homography
        Mat inv_homography = homography.inv();
        inv_homography.convertTo(inv_homography, CV_32F);
        cout << " ===== INV  homography ===== " << endl;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                cout << fixed << setw(8) << setprecision(3)
                     << inv_homography.at<float>(i, j) << " ";
            }
            cout << endl;
        }

        cuda_image_warping(
            img_right.data, stitch_img.data,
            hr, wr, stitch_height, stitch_width,
            (float *)inv_homography.data);

        cuda_linear_blending(img_left.data, stitch_img.data, stitch_img.data, stitch_height, stitch_width, wl);

        return removeBlackBorder(stitch_img);
    }

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

    Mat removeBlackBorder(const Mat &img)
    {
        int h = img.rows, w = img.cols;
        int reduced_h = h, reduced_w = w;

        // from right to left
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

        // from bottom to top
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

        // Check CUDA device
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)
        {
            cerr << "No CUDA devices found!" << endl;
            return -1;
        }

        cout << "Found " << deviceCount << " CUDA device(s)" << endl;

        string left_path = argv[1];
        string right_path = argv[2];
        string stitch = argv[3];

        Mat img_left = imread(left_path);
        Mat img_right = imread(right_path);

        if (img_left.empty() || img_right.empty())
        {
            cerr << "Could not load images!" << endl;
            return -1;
        }

        // Mat total;
        // hconcat(img_left, img_right, total);
        // imshow("Total", total);

        CudaStitcher stitcher;
        string blending_mode = "linearBlendingWithConstant";
        Mat result = stitcher.stitch(left_path, right_path, blending_mode, 0.75f);

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
