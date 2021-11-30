# 简介
用deepstream-test3跑yolov5，并对小汽车(car)进行流量统计
# 注意
这个程序只适用于Nvidia jetson盒子，并且在使用之前需要进行编译。

deepstream版本：5.0

YoloV5版本：5.0

首先，请保证你是jetson上进行编译的，然后请保证你的jetson安装了deepstream，其次，要把这个这个项目和deepstream里面的test例子项目放在同一个文件下，然后进行编译。

# 效果展示
见https://www.bilibili.com/video/BV1Xh41187tQ?from=search&seid=15658977398405425191
# 使用教程
【step1】生成程序需要用到的两个文件，一个是tensorRT要用的.engine文件，一个是libmyplugins.so文件，这两个文件可以从这个项目中生成：https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5

【step2】生成libnvdsinfer_custom_impl_Yolo.so文件,生成方式在这个项目里：https://github.com/DanaHan/Yolov5-in-Deepstream-5.0

【step3】配置config_infer_primary_yoloV5.txt文件，将libnvdsinfer_custom_impl_Yolo.so文件和.engine文件的路径都加到相应位置。

【step4】打开jetson终端，执行如下操作
```
git clone https://github.com/dongdongdong1217/yolov5-deepstream-test3.git
cd yolov5-deepstream-test3/
make
//下面这两个视频路径要根据具体的路径进行更改
LD_PRELOAD=./libmyplugins.so ./deepstream-test3-app file:///home/ubuntu/video1.mp4 或者LD_PRELOAD=./libmyplugins.so ./deepstream-test3-app rtsp://127.0.0.1/video2
```

# 说明
这是本人本科期间做的一个项目，当时学识有限，可能有些代码不是最高效的，但是保证能跑通且达到视频中的效果，仅供大家参考。

