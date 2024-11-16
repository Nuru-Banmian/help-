import cv2
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 加载预训练的DETR模型和特征提取器
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
TF_ENABLE_ONEDNN_OPTS=0
class TargetDetection:
    def __init__(self, modelpath):
        self.modelpath = modelpath  # 保存模型位置的
        pass

    def detection_crop_objects(self, image_paths: list[str]):
        features = []
        crop_info = {}

        for idx, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")  # 使用PIL加载图片

            # 特征提取
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # 解析预测结果
            target_sizes = torch.tensor([image.size][::-1])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

            labels = [model.config.id2label[label.item()] for label in results['labels']]
            boxes = results['boxes'].tolist()

            # 收集特征
            features.append((image_path, labels))

            # 收集裁剪信息
            boxes = [[round(coord) for coord in box] for box in boxes]
            crop_info[idx] = [(label, box) for label, box in zip(labels, boxes)]

        return features, crop_info

# 示例使用
if __name__ == "__main__":
    detector = TargetDetection(modelpath='./models')
    image_paths = ["1.jpg"]
    features, crop_info = detector.detection_crop_objects(image_paths)
    print("Features:", features)
    print("Crop Info:", crop_info)