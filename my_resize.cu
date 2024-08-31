#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "my_resize.h"


__device__ void bilinear_filter(float* src, int src_width, int src_height, 
                                float* dst, int dst_width, int dst_height,
                                int dst_x, int dst_y, int ch) {
    float* dst_pix_ptr = dst + (dst_y * dst_width + dst_x) * ch;
    float src_x_f = (dst_x + 0.5) * ((float) src_width / dst_width) - 0.5;
    float src_y_f = (dst_y + 0.5) * ((float) src_height / dst_height) - 0.5;
    src_x_f = min(src_width - 1., max(0., src_x_f));
    src_y_f = min(src_height - 1., max(0., src_y_f));

    int src_x = floor(src_x_f);
    int src_y = floor(src_y_f);

    float u = src_x_f - src_x;
    float v = src_y_f - src_y;

    float r_base = u * v;
    float r1 = 1 - u -v + r_base;
    float r2 = v - r_base;
    float r3 = u - r_base;
    float r4 = r_base;

    float* src_pix_ptr = src + (src_y * src_width + src_x) * ch;
    float* src_pix_ptr_next;
    float* src_pix_ptr_nextline;
    float* src_pix_ptr_nextline_next;

    if (src_x == src_width - 1) {
        src_pix_ptr_next = src_pix_ptr;
    } else {
        src_pix_ptr_next = src_pix_ptr + ch;
    }
    if (src_y == src_height - 1) {
        src_pix_ptr_nextline = src_pix_ptr;
        src_pix_ptr_nextline_next = src_pix_ptr_next;
    } else {
        src_pix_ptr_nextline = src_pix_ptr + src_width * ch;
        src_pix_ptr_nextline_next = src_pix_ptr_next + src_width * ch;
    }

    for (int i = 0; i < ch; i++) {
        dst_pix_ptr[i] = r1 * src_pix_ptr[i] + r2 * src_pix_ptr_nextline[i] + 
                        r3 * src_pix_ptr_next[i] + r4 * src_pix_ptr_nextline_next[i];
    }
}


__global__ void bilinear_resize_kernel(float* src_ptr_d, int src_width, int src_height, 
                                        float* dst_ptr_d, int dst_width, int dst_height, int ch) {
    int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int y_idx = threadIdx.y + blockDim.y * blockIdx.y;

    int x_stride = blockDim.x * gridDim.x;
    int y_stride = blockDim.y * gridDim.y;

    for (int i = x_idx; i < dst_width; i += x_stride) {
        for (int j = y_idx; j < dst_height; j += y_stride) {
            bilinear_filter(src_ptr_d, src_width, src_height, dst_ptr_d, dst_width, dst_height, i, j, ch);
        }
    }
}


void my_bilinear_resize(cv::Mat& src_img, cv::Mat& dst_img) {
    float *dst_ptr_d, *src_ptr_d;
    int src_width = src_img.cols;
    int src_height = src_img.rows;
    int dst_width = dst_img.cols;
    int dst_height = dst_img.rows;
    int ch = src_img.channels();

    cudaMalloc((void**)&src_ptr_d, (src_width*src_height*ch*sizeof(float)));
    cudaMalloc((void**)&dst_ptr_d, (dst_width*dst_height*ch*sizeof(float)));
    cudaMemcpy(src_ptr_d, src_img.data, (src_width*src_height*ch*sizeof(float)), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y);    
    bilinear_resize_kernel<<<gridSize, blockSize>>>(src_ptr_d, src_width, src_height, dst_ptr_d, dst_width, dst_height, ch);
    // 检查 CUDA 错误
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr!= cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl;
    }

    cudaMemcpy(dst_img.data, dst_ptr_d, (dst_width*dst_height*ch*sizeof(float)), cudaMemcpyDeviceToHost);

    cudaFree(src_ptr_d);
    cudaFree(dst_ptr_d);
}


__device__ void area_filter(float* src, int src_width, int src_height, 
                             float* dst, int dst_width, int dst_height,
                             int dst_x, int dst_y, int ch,
                             float* pix_sum_ptr) {
    float* dst_pix_ptr = dst + (dst_y * dst_width + dst_x) * ch;
    float dst_left = dst_x - 0.5;
    float dst_top = dst_y - 0.5;
    float dst_right = dst_x + 0.5;
    float dst_bot = dst_y + 0.5;

    float src_left = (dst_left + 0.5) * ((float) src_width / dst_width) - 0.5;
    float src_top = (dst_top + 0.5) * ((float) src_height / dst_height) - 0.5;
    float src_right = (dst_right + 0.5) * ((float) src_width / dst_width) - 0.5;
    float src_bot = (dst_bot + 0.5) * ((float) src_height / dst_height) - 0.5;

    float total_area = (src_bot - src_top) * (src_right - src_left);

    for (int i = 0; i < ch; i++) {
        pix_sum_ptr[i] = 0;
    }

    for (int m = floor(src_left + 0.5); m < floor(src_right + 0.5) + 1; m++) {
        for (int n = floor(src_top + 0.5); n < floor(src_bot + 0.5) + 1; n++) {
            float left_bound = max(src_left + 0.5, float(m));
            float right_bound = min(src_right + 0.5, float(m + 1));
            float top_bound = max(src_top + 0.5, float(n));
            float bot_bound = min(src_bot + 0.5, float(n + 1));
            
            float* src_pix_ptr = src + (n * src_width + m) * ch;
            
            for (int i = 0; i < ch; i++) {
                pix_sum_ptr[i] += (right_bound - left_bound) * (bot_bound - top_bound) * src_pix_ptr[i];
            }
        }
    }

    for (int i = 0; i < ch; i++) {
        dst_pix_ptr[i] = pix_sum_ptr[i] / total_area;
    }
}


__global__ void area_resize_kernel(float* src_ptr_d, int src_width, int src_height, 
                                    float* dst_ptr_d, int dst_width, int dst_height, int ch,
                                    float* k_all_pix_sum_ptr) {
    int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int y_idx = threadIdx.y + blockDim.y * blockIdx.y;

    int x_stride = blockDim.x * gridDim.x;
    int y_stride = blockDim.y * gridDim.y;
    float *k_pix_sum_ptr = k_all_pix_sum_ptr + (y_idx * x_stride + x_idx) * ch;

    for (int i = x_idx; i < dst_width; i += x_stride) {
        for (int j = y_idx; j < dst_height; j += y_stride) {
            area_filter(src_ptr_d, src_width, src_height, dst_ptr_d, dst_width, dst_height, i, j, ch, k_pix_sum_ptr);
        }
    }
}


void my_area_resize(cv::Mat& src_img, cv::Mat& dst_img) {
    float *dst_ptr_d, *src_ptr_d;
    int src_width = src_img.cols;
    int src_height = src_img.rows;
    int dst_width = dst_img.cols;
    int dst_height = dst_img.rows;
    int ch = dst_img.channels();

    cudaMalloc((void**)&src_ptr_d, (src_width*src_height*ch*sizeof(float)));
    cudaMalloc((void**)&dst_ptr_d, (dst_width*dst_height*ch*sizeof(float)));
    cudaMemcpy(src_ptr_d, src_img.data, (src_width*src_height*ch*sizeof(float)), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y);    

    float *k_all_pix_sum_ptr;
    
    cudaMalloc((void**)&k_all_pix_sum_ptr, blockSize.x * gridSize.x * blockSize.y * gridSize.y * ch * sizeof(float));

    area_resize_kernel<<<gridSize, blockSize>>>(src_ptr_d, src_width, src_height, dst_ptr_d, dst_width, dst_height, ch, k_all_pix_sum_ptr);
    // 检查 CUDA 错误
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr!= cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl;
    }

    cudaMemcpy(dst_img.data, dst_ptr_d, (dst_width*dst_height*ch*sizeof(float)), cudaMemcpyDeviceToHost);
    
    cudaFree(src_ptr_d);
    cudaFree(dst_ptr_d);
    cudaFree(k_all_pix_sum_ptr);
}


__device__ void dFilter(float *src, int src_width, int src_height, 
                        float *dst, int dst_width, int dst_height,
                        int ch, int dst_x, int dst_y, 
                        int xmin, int xmax, int ymin, int ymax,
                        float *k_x, float *k_y,
                        float ww_x, float ww_y) {
    float *src_pix_ptr;
    float *dst_pix_ptr;

    for (int c = 0; c < ch; c++) {
        float ss = 0.0;
        for (int y = ymin; y < ymax; y++) {
            for (int x = xmin; x < xmax; x++) {
                src_pix_ptr = src + (y * src_width + x) * ch + c;
                ss = ss + (*src_pix_ptr) * k_y[y - ymin] * k_x[x - xmin];
            }
        }
        dst_pix_ptr = dst + (dst_y * dst_width + dst_x) * ch + c;
        *dst_pix_ptr = ss * ww_x * ww_y;
    }
}


__device__ void antialias_filter(float x, float &w) {
    /* lanczos (truncated sinc) */
    if (-3.0 <= x && x < 3.0) {
        if (x == 0.0) {
            w = 1.0;
        }
        else {
            w = 3 * sin(x * M_PI) * sin(x * M_PI / 3) / (x * x * M_PI * M_PI);
        }
    }
    else
        w = 0.0;

}


__global__ void lanczos_rsz_kernel(float *src_ptr_d, int src_width, int src_height,
                                    float *dst_ptr_d, int dst_width, int dst_height,
                                    int ch, float scale_x, float scale_y,
                                    float filterscale_x, float filterscale_y,
                                    float support_x, float support_y,
                                    float *k_x_d, float *k_y_d) {
    int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int y_idx = threadIdx.y + blockDim.y * blockIdx.y;

    int x_stride = blockDim.x * gridDim.x;
    int y_stride = blockDim.y * gridDim.y;

    float center_x, center_y, ww_x, ww_y, ss_x, ss_y, ymin, ymax, xmin, xmax;

    float *k_x = k_x_d + (y_idx * x_stride + x_idx) * ((int)support_x * 2 + 2);
    float *k_y = k_y_d + (y_idx * x_stride + x_idx) * ((int)support_y * 2 + 2);

    for (int xx = x_idx; xx < dst_width; xx += x_stride) {
        for (int yy = y_idx; yy < dst_height; yy += y_stride) {
            center_y = (yy + 0.5) * scale_y;
            ww_y = 0.0;
            ss_y = 1.0 / filterscale_y;
            center_x = (xx + 0.5) * scale_x;
            ww_x = 0.0;
            ss_x = 1.0 / filterscale_x;
            
            /* calculate filter_y weights */
            ymin = floor(center_y - support_y);
            if (ymin < 0.0)
                ymin = 0.0;
            ymax = ceil(center_y + support_y);
            if (ymax > (float) src_height)
                ymax = (float) src_height;
            for (int y = (int) ymin; y < (int) ymax; y++) {
                float w = 0.0;
                antialias_filter((y - center_y + 0.5) * ss_y, w);
                w = w * ss_y;
                k_y[y - (int) ymin] = w;
                ww_y = ww_y + w;
            }
            if (ww_y == 0.0)
                ww_y = 1.0;
            else
                ww_y = 1.0 / ww_y;
            
            /* calculate filter_x weights */
            xmin = floor(center_x - support_x);
            if (xmin < 0.0)
                xmin = 0.0;
            xmax = ceil(center_x + support_x);
            if (xmax > (float) src_width)
                xmax = (float) src_width;
            for (int x = (int) xmin; x < (int) xmax; x++) {
                float w = 0.0;
                antialias_filter((x - center_x + 0.5) * ss_x, w);
                w = w * ss_x;
                k_x[x - (int) xmin] = w;
                ww_x = ww_x + w;
            }
            if (ww_x == 0.0)
                ww_x = 1.0;
            else
                ww_x = 1.0 / ww_x;
            
            dFilter(src_ptr_d, src_width, src_height, dst_ptr_d, dst_width, dst_height, ch, xx, yy,
                        (int)xmin, (int)xmax, (int)ymin, (int)ymax, k_x, k_y, ww_x, ww_y);

        }
    }
}

void my_lanczos_resize(cv::Mat &src_img, cv::Mat &dst_img) {
    float *dst_ptr_d, *src_ptr_d;
    int src_width = src_img.cols;
    int src_height = src_img.rows;
    int dst_width = dst_img.cols;
    int dst_height = dst_img.rows;
    int ch = src_img.channels();
    
    float support_x, support_y, scale_x, scale_y, filterscale_x, filterscale_y;

    cudaMalloc((void**)&src_ptr_d, (src_width * src_height * ch * sizeof(float)));
    cudaMalloc((void**)&dst_ptr_d, (dst_width * dst_height * ch * sizeof(float)));
    cudaMemcpy(src_ptr_d, src_img.data, (src_width * src_height * ch * sizeof(float)), cudaMemcpyHostToDevice);

    filterscale_x = scale_x = (float) src_width / dst_width;
    filterscale_y = scale_y = (float) src_height / dst_height;

    support_x = support_y = 3.0; // support of ANTIALIAS.

    if (filterscale_x < 1.0) {
        filterscale_x = 1.0;
        support_x = 0.5;
    }
    support_x = support_x * filterscale_x;

    if (filterscale_y < 1.0) {
        filterscale_y = 1.0;
        support_y = 0.5;
    }
    support_y = support_y * filterscale_y;

    dim3 blockSize(32, 32);
    dim3 gridSize((dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y);

    float *k_x_d, *k_y_d;
    
    cudaMalloc((void**)&k_x_d, blockSize.x * gridSize.x * blockSize.y * gridSize.y * ((int)support_x * 2 + 2) * sizeof(float));
    cudaMalloc((void**)&k_y_d, blockSize.x * gridSize.x * blockSize.y * gridSize.y * ((int)support_y * 2 + 2) * sizeof(float));

    lanczos_rsz_kernel<<<gridSize, blockSize>>>(src_ptr_d, src_width, src_height, dst_ptr_d, dst_width, dst_height, ch,
                                            scale_x, scale_y, filterscale_x, filterscale_y, support_x, support_y, k_x_d, k_y_d);
    // 检查 CUDA 错误
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr!= cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl;
    }

    cudaMemcpy(dst_img.data, dst_ptr_d, (dst_width * dst_height * ch * sizeof(float)), cudaMemcpyDeviceToHost);

    cudaFree(src_ptr_d);
    cudaFree(dst_ptr_d);

    cudaFree(k_x_d);
    cudaFree(k_y_d);
}
