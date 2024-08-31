#include "opencv2/opencv.hpp"
#include "my_resize.h"
#include <iostream>

using namespace std;


int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "Please specify image path!" << endl;
        cout << "Usage: ./test img_path resize_ratio" << endl;
        return -1;
    }

    cv::Mat src_mat = cv::imread(argv[1]);
    if (!src_mat.data) {
        cout << "Error reading " << argv[1] << endl;
        return -1;
    }
    
    float scale = atof(argv[2]);
    if (scale <= 0) {
        cout << "resize ratio must > 0.";
        return -1;
    }
    int resize_w = int(scale * src_mat.cols + 0.5);
    int resize_h = int(scale * src_mat.rows + 0.5);

    src_mat.convertTo(src_mat, CV_32FC3);
    cout << "src image WxH: " << src_mat.cols << "x" << src_mat.rows << endl;
    cout << "dst image WxH: " << resize_w << "x" << resize_h << endl;

    cv::Mat rsz;
    cv::resize(src_mat, rsz, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_AREA);
    cv::imwrite("cv_area.png", rsz);
    cv::resize(src_mat, rsz, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
    cv::imwrite("cv_linear.png", rsz);
    cv::resize(src_mat, rsz, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_NEAREST);
    cv::imwrite("cv_nearest.png", rsz);
    cv::resize(src_mat, rsz, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LANCZOS4);
    cv::imwrite("cv_lanczos4.png", rsz);
    cv::resize(src_mat, rsz, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_CUBIC);
    cv::imwrite("cv_cubic.png", rsz);

    cv::Mat my_rsz_mat(resize_h, resize_w, CV_32FC3);
    my_area_resize(src_mat, my_rsz_mat);
    cv::imwrite("my_area.png", my_rsz_mat);

    my_bilinear_resize(src_mat, my_rsz_mat);
    cv::imwrite("my_linear.png", my_rsz_mat);
    
    my_lanczos_resize(src_mat, my_rsz_mat);
    cv::imwrite("my_lanczos.png", my_rsz_mat);

    return 0;
}
