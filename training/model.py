# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_config import (
    NUMBER_OF_CLASSES,
    GRID_SIZE,
    NUMBER_OF_BBOXES,
    IMAGE_SIZE,
    CONF_THRESH,
    IOU_THRESH
)


class YOLO(nn.Module):
    def __init__(self, num_classes=NUMBER_OF_CLASSES):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.backbone = self._create_backbone()
        self.head = self._create_head()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def _create_backbone(self):
        layers = []
        # Layer 1
        layers.append(nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # Layer 2
        layers.append(nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # Layer 3
        layers.append(nn.Conv2d(in_channels=192,
                      out_channels=128, kernel_size=1))
        layers.append(nn.LeakyReLU(0.1))
        # Layer 4
        layers.append(nn.Conv2d(in_channels=128,
                      out_channels=256, kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(0.1))
        # Layer 5
        layers.append(nn.Conv2d(in_channels=256,
                      out_channels=256, kernel_size=1))
        layers.append(nn.LeakyReLU(0.1))
        # Layer 6
        layers.append(nn.Conv2d(in_channels=256,
                      out_channels=512, kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # Layers 7-14
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels=512,
                          out_channels=256, kernel_size=1))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Conv2d(in_channels=256,
                          out_channels=512, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.1))
        # Layer 15
        layers.append(nn.Conv2d(in_channels=512,
                      out_channels=1024, kernel_size=1))
        layers.append(nn.LeakyReLU(0.1))
        # Layer 16
        layers.append(nn.Conv2d(in_channels=1024,
                      out_channels=1024, kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # Layers 17-20
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels=1024,
                          out_channels=512, kernel_size=1))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Conv2d(in_channels=512,
                          out_channels=1024, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.1))
        return nn.Sequential(*layers)

    def _create_head(self):
        layers = []
        # Layer 21
        layers.append(nn.Conv2d(in_channels=1024,
                      out_channels=1024, kernel_size=3))
        layers.append(nn.LeakyReLU(0.1))
        # Layer 22
        layers.append(nn.Conv2d(in_channels=1024, out_channels=1024,
                      kernel_size=3, stride=2, padding=2))
        layers.append(nn.LeakyReLU(0.1))
        # Layers 23-24
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels=1024,
                          out_channels=1024, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(1024 * 7 * 7, 4096))
        layers.append(nn.Dropout())
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Linear(4096, GRID_SIZE * GRID_SIZE *
                      (NUMBER_OF_BBOXES * 5 + self.num_classes)))
        return nn.Sequential(*layers)

    def detect(self, images, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
        outputs = self(images)
        return non_max_suppression(outputs, conf_thresh, iou_thresh)

# Helper functions for detection


def non_max_suppression(predictions, conf_thresh, iou_thresh):
    nms_predictions = []
    batch_size = predictions.size(0)
    predictions = predictions.view(
        batch_size, NUMBER_OF_BBOXES * 5 + NUMBER_OF_CLASSES, GRID_SIZE, GRID_SIZE).permute(0, 2, 3, 1)
    for image in range(batch_size):
        image_predictions = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                class_probs = predictions[image, y, x, 5 * NUMBER_OF_BBOXES:]
                class_score, class_id = torch.max(class_probs, 0)
                for bbox in range(NUMBER_OF_BBOXES):
                    box_data = predictions[image, y, x, bbox *
                                           5:(bbox + 1) * 5 + NUMBER_OF_CLASSES]
                    x_center, y_center, width, height = F.relu(box_data[:4])
                    objectness = torch.sigmoid(box_data[4])
                    if objectness < conf_thresh:
                        continue
                    x_center = (x + x_center) * (IMAGE_SIZE / GRID_SIZE)
                    y_center = (y + y_center) * (IMAGE_SIZE / GRID_SIZE)
                    width = width * IMAGE_SIZE
                    height = height * IMAGE_SIZE
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    image_predictions.append(
                        [x1, y1, x2, y2, objectness, class_score, class_id])
        image_predictions = sorted(
            image_predictions, key=lambda x: x[4], reverse=True)
        nms_predictions.append(non_max_suppression_per_image(
            image_predictions, iou_thresh))
    return nms_predictions


def non_max_suppression_per_image(image_predictions, iou_thresh):
    nms_predictions = []
    while image_predictions:
        best_box = image_predictions.pop(0)
        nms_predictions.append(best_box)
        iou = iou_fn(best_box[0:4], [box[0:4] for box in image_predictions])
        image_predictions = [box for i, box in enumerate(
            image_predictions) if iou[i] < iou_thresh]
    return nms_predictions


def iou_fn(box1, boxes):
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    inter_area = torch.tensor(
        [intersection_area(box1, box2) for box2 in boxes])
    union_area = box1_area + \
        torch.tensor([(box2[2] - box2[0]) * (box2[3] - box2[1])
                     for box2 in boxes]) - inter_area
    return inter_area / union_area


def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(box1, box2):
    box1_x1 = box1[..., 0] - box1[..., 2] / 2
    box1_y1 = box1[..., 1] - box1[..., 3] / 2
    box1_x2 = box1[..., 0] + box1[..., 2] / 2
    box1_y2 = box1[..., 1] + box1[..., 3] / 2

    box2_x1 = box2[..., 0] - box2[..., 2] / 2
    box2_y1 = box2[..., 1] - box2[..., 3] / 2
    box2_x2 = box2[..., 0] + box2[..., 2] / 2
    box2_y2 = box2[..., 1] + box2[..., 3] / 2

    inter_rect_x1 = torch.max(box1_x1, box2_x1)
    inter_rect_y1 = torch.max(box1_y1, box2_y1)
    inter_rect_x2 = torch.min(box1_x2, box2_x2)
    inter_rect_y2 = torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / torch.clamp(union_area, min=1e-10)
