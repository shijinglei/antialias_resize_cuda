# antialias_resize_cuda
This code contains cuda implementation of bilinear resize, area resize and antialias resize.
## compile
```
mkdir build && cd build
cmake ..
make -j
```

## test
```
./test ../test_imgs/car.png 0.23
```
This command sets the image below as input image, and the resize ratio is 0.23  
![image](test_imgs/car.png)   
The result image using opencv bilinear resize is  
![image](build/cv_linear.png)   
whereas our result image using antialias resize is  
![image](build/my_lanczos.png)

