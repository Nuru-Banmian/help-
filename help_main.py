import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrImageProcessor, DetrForObjectDetection

# 加载预训练的DETR模型和特征提取器
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

class TargetDetection:
    def __init__(self, modelpath):
        self.modelpath = modelpath  # 保存模型位置的

    def detect_and_visualize(self, image_paths: list[str], output_path: str):
        features = []
        crop_info = {}

        for idx, image_path in enumerate(image_paths):
            image = Image.open(image_path)

            # 特征提取
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # 解析预测结果
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.2)[0]
            labels = [model.config.id2label[label.item()] for label in results['labels']]

            features.append((image_path, labels))

            # 收集裁剪信息
            crop_info[idx] = [(label, box) for label, box in zip(labels, results['boxes'])]

            # 绘制检测框
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(image)
            for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
                box = box.tolist()
                label = model.config.id2label[label.item()]
                ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2))
                ax.text(box[0], box[1], f'{label}: {score:.2f}', fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

            # 保存结果图像
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        return features, crop_info


# 示例使用
if __name__ == "__main__":
    # 使用 TargetDetection 类
    detector = TargetDetection(modelpath='./models')
    image_paths = ["5.jpg"]
    output_dir = 'output_image.jpg'
    features, crop_info = detector.detect_and_visualize(image_paths, output_dir)
    print("Features:", features)
    print("Crop Info:", crop_info)