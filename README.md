# 简介
用deepstream-test3跑yolov5，并对小汽车(car)进行流量统计
# 注意
这个程序只适用于Nvidia jetson盒子，并且在使用之前需要进行编译。
# 效果展示
略
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
LD_PRELOAD=./libmyplugin.so ./deepstream-test3-app file:///home/ubuntu/video1.mp4 或者LD_PRELOAD=./libmyplugin.so ./deepstream-test3-app rtsp://127.0.0.1/video2

