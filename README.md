# help-


markdown
深色版本
# 目标检测项目

## 项目概述

这是一个基于 DETR (DEtection TRansformer) 模型的目标检测项目。该项目使用了 Hugging Face 的 `transformers` 库中的预训练模型 `facebook/detr-resnet-50`，实现了对输入图像中目标的检测，并返回检测到的对象及其边界框信息。

## 技术栈

- **Python**: 主要编程语言。
- **PyTorch**: 深度学习框架，用于模型的加载和推理。
- **PIL (Pillow)**: 图像处理库，用于加载和处理图像。
- **Transformers**: Hugging Face 提供的库，用于加载和使用预训练模型。
- **OpenCV (cv2)**: 图像处理库，虽然在这个项目中没有直接使用，但可以用于后续的图像处理任务。
- **Matplotlib**: 数据可视化库，用于绘制检测结果。

## 安装依赖

在开始使用项目之前，请确保安装了以下依赖：

```sh
pip install torch torchvision transformers pillow opencv-python
```
使用方法
1. 下载预训练模型
项目已经预先加载了`facebook/detr-resnet-50`模型，因此无需手动下载。

2. 修改图像路径
  在`main`函数中，修改`image_paths`列表，添加你想要检测的图像路径。例如：
``` python
image_paths = ["1.jpg", "2.jpg"]
```

3. 运行代码
运行 main 函数，执行目标检测：
``` python
if __name__ == "__main__":
    detector = TargetDetection(modelpath='./models')
    image_paths = ["1.jpg"]
    features, crop_info = detector.detection_crop_objects(image_paths)
    print("Features:", features)
    print("Crop Info:", crop_info)
```

4. 查看结果
代码将输出检测到的对象及其边界框信息。输出格式如下：

- `features`: 列表，每个元素是一个元组 (image_path, labels)，其中 image_path 是图像的路径，labels 是检测到的对象列表。
- `crop_info`: 字典，键是图像在 image_paths 列表中的索引，值是一个列表，每个列表元素是一个元组 (label, box)，其中 label 是检测到的对象类别，box 是对象的边界框坐标 (x_min, y_min, x_max, y_max)。

示例输出
假设`image_paths`是`["1.jpg", "2.jpg"]`，处理完两张图片后，输出可能如下：
```python
Features: [
    ("1.jpg", ["person", "car", "bicycle"]),
    ("2.jpg", ["dog", "cat"])
]

Crop Info: {
    0: [("person", [100, 150, 200, 250]), ("car", [300, 350, 400, 450]), ("bicycle", [500, 550, 600, 650])],
    1: [("dog", [10, 20, 30, 40]), ("cat", [50, 60, 70, 80])]
}
