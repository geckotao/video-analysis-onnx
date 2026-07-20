视频目标检测截图工具onnx模型版

CUDA加速版源码：https://github.com/geckotao/video-analysis/ 

注：此版本程序为非NVIDIA显卡设计，兼容 DirectX 12 的GPU（包括集成显卡，操作系统Windows10 1709+）就可使用GPU加速，有NVIDIA显卡请用CUDA加速版程序更优。


功能简介

本工具可对一个或多个视频文件进行目标检测（基于yolo26x.onnx 模型）自动截图指定类别的目标，并支持以下功能：

选择多个目标类别（如“人”、“自行车”、“小车”等YOLO的80种目标）

设置关注区域（ROI）：只在指定区域内检测目标

移动目标检测：仅对首次出现或发生移动的目标截图

实时预览处理画面（带检测框和 ROI 显示）

自动跳帧处理（提升速度）

支持在截图上标注目标类别与置信度

支持中文路径与文件名

依赖

pip install onnxruntime-directml opencv-python pyside6  numpy configparser

