#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

                                                        
void my_bilinear_resize(cv::Mat& src_img, cv::Mat& dst_img);


void my_area_resize(cv::Mat& src_img, cv::Mat& dst_img);


void my_lanczos_resize(cv::Mat& src_img, cv::Mat& dst_img);
