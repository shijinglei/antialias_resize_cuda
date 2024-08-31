CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

project(Test LANGUAGES CXX CUDA)

# set(OpenCV_DIR /home/shijinglei/projects/opencv/opencv-install/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

add_executable(test main.cpp my_resize.cu)

target_include_directories(test PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS} ${CUDA_LIBRARIES_DIR})
message("CUDA_LIBRARY_DIRS: " ${CUDA_INCLUDE_DIRS})
target_link_libraries(test ${OpenCV_LIBS} ${CUDA_LIBRARIES})
