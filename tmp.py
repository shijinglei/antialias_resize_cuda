from unicodedata import decimal
import cv2
from math import floor
import numpy as np
from numba import cuda


@cuda.jit
def my_bilinear_resize(src, dst):
    src_width = src.shape[1]
    src_height = src.shape[0]
    dst_width = dst.shape[1]
    dst_height = dst.shape[0]
    j, i = cuda.grid(2)
    stride_j, stride_i = cuda.gridsize(2)
    for x in range(i, dst.shape[1], stride_i):
        for y in range(j, dst.shape[0], stride_j):
            src_x_f = (x + 0.5) * (float(src_width) / dst_width) - 0.5
            src_y_f = (y + 0.5) * (float(src_height) / dst_height) - 0.5
            src_x_f = min(src_width - 1, max(0, src_x_f))
            src_y_f = min(src_height - 1, max(0, src_y_f))

            src_x = floor(src_x_f)
            src_y = floor(src_y_f)

            u = src_x_f - src_x
            v = src_y_f - src_y

            r_base = u * v
            r1 = 1 - u - v + r_base
            r2 = v - r_base
            r3 = u - r_base
            r4 = r_base

            src_left = int(src_x)
            src_top = int(src_y)
            if src_x == src_width:
                src_right = int(src_x)
            else:
                src_right = int(src_x + 1)
            if src_y == src_height:
                src_bot = int(src_y)
            else:
                src_bot = int(src_y + 1)

            top_left = src[src_top][src_left]
            top_right = src[src_top][src_right]
            bot_left = src[src_bot][src_left]
            bot_right = src[src_bot][src_right]
            dst[y][x][0] = r1 * top_left[0] + r2 * bot_left[0] + r3 * top_right[0] + r4 * bot_right[0]
            dst[y][x][1] = r1 * top_left[1] + r2 * bot_left[1] + r3 * top_right[1] + r4 * bot_right[1]
            dst[y][x][2] = r1 * top_left[2] + r2 * bot_left[2] + r3 * top_right[2] + r4 * bot_right[2]


@cuda.jit
def my_area_resize(src, dst):
    j, i = cuda.grid(2)
    stride_j, stride_i = cuda.gridsize(2)
    for x in range(i, dst.shape[1], stride_i):
        for y in range(j, dst.shape[0], stride_j):
            dst[y][x] = 1
            # dst[y][x][0] = x
            # dst[y][x][1] = y
            # dst[y][x][2] = 1
            dst_left = x - 0.5
            dst_top = y - 0.5
            dst_right = x + 0.5
            dst_bot = y + 0.5

            src_left = (dst_left + 0.5) * float(src.shape[1]) / dst.shape[1] - 0.5
            src_top = (dst_top + 0.5) * float(src.shape[0]) / dst.shape[0] - 0.5
            src_right = (dst_right + 0.5) * float(src.shape[1]) / dst.shape[1] - 0.5
            src_bot = (dst_bot + 0.5) * float(src.shape[0]) / dst.shape[0] - 0.5

            total_area = (src_bot - src_top) * (src_right - src_left)
            pix_weighted_sum_0 = 0.
            pix_weighted_sum_1 = 0.
            pix_weighted_sum_2 = 0.

            for m in range(floor(src_left + 0.5), floor(src_right + 0.5) + 1):
                for n in range(floor(src_top + 0.5), floor(src_bot + 0.5) + 1):
                    left_bound = max(src_left + 0.5, m)
                    right_bound = min(src_right + 0.5, m + 1)
                    top_bound = max(src_top + 0.5, n)
                    bot_bound = min(src_bot + 0.5, n + 1)
                    pix_weighted_sum_0 += (right_bound - left_bound) * \
                                          (bot_bound - top_bound) * src[n][m]
                    # pix_weighted_sum_0 += src[n][m][0]*(right_bound - left_bound) * (bot_bound - top_bound)#* src[n][m][0]
                    # pix_weighted_sum_1 += src[n][m][1]*(right_bound - left_bound) * (bot_bound - top_bound)#* src[n][m][1]
                    # pix_weighted_sum_2 += src[n][m][2]*(right_bound - left_bound) * (bot_bound - top_bound)#* src[n][m][2]
            #
            dst[y][x] = pix_weighted_sum_0 / total_area
            # dst[y][x][0] = float(pix_weighted_sum_0) #/ total_area
            # dst[y][x][1] = float(pix_weighted_sum_1) #/ total_area
            # dst[y][x][2] = float(pix_weighted_sum_2) #/ total_area


img = cv2.imread("0.png").astype(np.float32)
# img = np.random.random((15, 10, 3)).astype(np.float32)

# img = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.float32)
# img = np.stack((img, img, img, img))
# img = np.stack((img, img, img), axis=-1)

rsz = cv2.resize(img, (300, 220), interpolation=cv2.INTER_AREA)
# print(rsz)

my_rsz = np.zeros_like(rsz)
rsz = np.round(np.clip(rsz, 0, 255), decimals=1).astype(np.uint8)
cv2.imwrite("cv_area.png", rsz)

src_dev = cuda.to_device(img)
dst_dev = cuda.to_device(my_rsz)

blocks = (10, 10)
threads_per_block = (32, 32)
my_area_resize[blocks, threads_per_block](src_dev[:,:,0], dst_dev[:,:,0])
my_area_resize[blocks, threads_per_block](src_dev[:,:,1], dst_dev[:,:,1])
my_area_resize[blocks, threads_per_block](src_dev[:,:,2], dst_dev[:,:,2])
# print(dst_dev.copy_to_host())
my_rsz = dst_dev.copy_to_host()
my_rsz = np.round(np.clip(my_rsz, 0, 255), decimals=1).astype(np.uint8)
cv2.imwrite("my_area.png", my_rsz)
print((my_rsz == rsz).all())
print(np.where(my_rsz != rsz))
