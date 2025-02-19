#!/usr/bin/env python

import argparse
import glob
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from accelerate import Accelerator

# Import SM Debugger for PyTorch
import smdebug.pytorch as smd

NUMBER_OF_CLASSES = 20  # VOC has 20 classes
GRID_SIZE = 7           # 7x7 grid
NUMBER_OF_BBOXES = 1     # One bounding box per cell
IMAGE_SIZE = 448

# Optimization hyperparameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.90
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Loss hyperparameters
LAMBDA_COORD = 50
LAMBDA_NOOB = 0.5
LAMBDA_OBJ = 10
LAMBDA_CLASS = 1

# Detection thresholds
CONF_THRESH = 0.5
IOU_THRESH = 0.5

# VOC labels
VOC_LABELS = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
    'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
    'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
    'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

# Derived constants
CLASS_LABELS = list(VOC_LABELS.keys())


def collate_fn(batch):
    """
    Custom collate function for the data loader that handles multiple objects in the same grid cell
    without overwriting bounding boxes.
    """
    images = []
    targets = []

    grid_size = GRID_SIZE
    num_boxes = NUMBER_OF_BBOXES
    num_classes = NUMBER_OF_CLASSES
    image_size = IMAGE_SIZE  # Assuming a square image

    for img, target in batch:
        images.append(img)
        target_tensor = torch.zeros(
            [num_boxes * 5 + num_classes, grid_size, grid_size])

        x_center_array, y_center_array, width_array, height_array, names = target.T

        # Track which boxes have been filled in each cell
        filled_boxes = torch.zeros(
            (grid_size, grid_size, num_boxes), dtype=torch.int64)

        for idx in range(len(x_center_array)):
            x_center = x_center_array[idx]
            y_center = y_center_array[idx]
            width = width_array[idx]
            height = height_array[idx]
            label_index = int(names[idx])

            # Determine grid cell indices
            grid_x = int(x_center * grid_size)
            grid_y = int(y_center * grid_size)

            # X and Y coordinates relative to the grid cell
            x_coord = x_center * grid_size - grid_x
            y_coord = y_center * grid_size - grid_y

            # Box coordinates relative to the grid cell
            box_coordinates = torch.tensor(
                [x_coord, y_coord, width, height])

            # Assign to the first available bounding box in this cell
            for b in range(num_boxes):
                if filled_boxes[grid_y, grid_x, b] == 0:
                    box_index = b * 5  # 5 parameters per box
                    target_tensor[box_index:box_index + 4,
                                  grid_y, grid_x] = box_coordinates
                    target_tensor[box_index + 4, grid_y,
                                  grid_x] = 1  # Objectness score
                    filled_boxes[grid_y, grid_x, b] += 1
                    break

            # Shared class probabilities for this grid cell
            # Reset class probabilities
            target_tensor[num_boxes * 5:, grid_y, grid_x] = 0
            target_tensor[num_boxes * 5 + label_index,
                          grid_y, grid_x] = 1  # Set class probability

        targets.append(target_tensor)

    images = torch.stack(images)
    targets = torch.stack(targets)

    return images, targets


class PreprocessedVOCDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory containing preprocessed .npz files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, '*.npz'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        image = data['image']  # Expected shape: (H, W, 3)
        # Expected shape: (N, 5) where each row is [x_center, y_center, width, height, class_id]
        boxes = data['boxes']

        # Convert image (NumPy array) to a PIL image for transformation
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor without normalization.
            image = transforms.ToTensor()(image)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        return image, boxes


def get_datasets(base_dir):
    """
    Assumes base_dir contains three subdirectories: 'train', 'val', and 'test',
    each holding the corresponding preprocessed .npz files.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PreprocessedVOCDataset(
        os.path.join(base_dir, 'train'), transform=transform)
    val_dataset = PreprocessedVOCDataset(
        os.path.join(base_dir, 'val'), transform=transform)
    test_dataset = PreprocessedVOCDataset(
        os.path.join(base_dir, 'test'), transform=transform)
    return train_dataset, val_dataset, test_dataset


def get_data_loaders(data_dir, batch_size=BATCH_SIZE, num_workers=4):
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# train_utils.py


def train_model(model, train_loader, validation_loader, model_output_dir, accelerator, num_epochs=10, loss_function=None, hook=None):
    device = accelerator.device
    print(f"Training on device '{device}'.")
    if loss_function is None:
        raise ValueError("A loss function must be provided.")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.90)
    model.to(device)

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        # Set debugger mode for training
        if hook is not None:
            hook.set_mode(smd.modes.TRAIN)
        total_loss = 0

        # Adjust learning rate schedule
        # for g in optimizer.param_groups:
        #     if epoch > 0 and epoch <= 5:
        #         g['lr'] = 0.00001 + (0.00009/5) * epoch
        #     elif epoch <= 80 and epoch > 5:
        #         g['lr'] = 0.0001
        #     elif epoch <= 110 and epoch > 80:
        #         g['lr'] = 0.00001
        #     elif epoch > 110:
        #         g['lr'] = 0.000001

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()

            # Record the training loss with SM Debugger.
            if hook is not None:
                hook.record_tensor_value("train_loss", loss)

            item_loss = loss.item()
            total_loss += item_loss
            progress_bar.set_postfix(train_loss=item_loss)
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = validate(model, validation_loader,
                                loss_function, hook)
        val_losses.append(avg_val_loss)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Save model if validation loss improves.
        if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
            model_path = os.path.join(
                model_output_dir, f'model_{int(time.time())}.pth')
            accelerator.save(model.state_dict(), model_path)
    return model


def validate(model, validation_loader, loss_function, device, hook=None):
    model.eval()
    if hook is not None:
        hook.set_mode(smd.modes.EVAL)
    total_loss = 0
    progress_bar = tqdm(validation_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = loss_function(outputs, targets)
            item_loss = loss.item()
            total_loss += item_loss

            # Record the validation loss with SM Debugger.
            if hook is not None:
                hook.record_tensor_value("val_loss", loss)
            progress_bar.set_postfix(val_loss=item_loss)
    return total_loss / len(validation_loader)


def calculate_ap_per_class(tp, conf, pred_cls, target_cls):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls)
    ap_dict = {}
    for c in unique_classes:
        pred_for_class = pred_cls == c
        target_for_class = target_cls == c
        if np.sum(target_for_class) == 0:
            ap_dict[c] = 0.0
            continue
        ap = average_precision_score(target_for_class, conf[pred_for_class])
        ap_dict[c] = ap
    return ap_dict


def flatten_bboxes(tensor):
    A, B, C, D = tensor.shape
    tensor = tensor.reshape(A, B, C * D).transpose(1, 2).reshape(-1, B)
    return tensor


def calculate_iou(boxes1, boxes2):
    boxes1 = flatten_bboxes(boxes1)
    boxes2 = flatten_bboxes(boxes2)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
        torch.clamp(inter_y2 - inter_y1, min=0)
    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / union_area
    return iou


def match_predictions(iou, iou_thresh):
    matches = []
    N, M = iou.shape
    for i in range(N):
        best_match = -1
        best_iou = iou_thresh
        for j in range(M):
            if iou[i, j] > best_iou:
                best_match = j
                best_iou = iou[i, j]
        if best_match != -1:
            matches.append((i, best_match))
    return matches


def update_metrics(matches, scores, pred_labels, gt_labels):
    tp = np.zeros(len(scores))
    fp = np.zeros(len(scores))
    for i, match in enumerate(matches):
        if pred_labels[i] == gt_labels[match]:
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp, scores, pred_labels


def decode_ground_truth(target):
    """
    Decode ground truth boxes and labels from the target tensor.
    Assumes target shape is [25, GRID_SIZE, GRID_SIZE] (5 for bbox + 20 for classes).
    """
    gt_boxes = []
    gt_labels = []
    cell_size = IMAGE_SIZE / GRID_SIZE
    # Loop over each grid cell
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Check if an object exists in this cell (objectness score is 1)
            if target[4, i, j] == 1:
                # Get box coordinates relative to the cell
                x_cell = target[0, i, j]
                y_cell = target[1, i, j]
                w = target[2, i, j]
                h = target[3, i, j]
                # Convert to absolute coordinates
                x_center = (j + x_cell) * cell_size
                y_center = (i + y_cell) * cell_size
                x1 = x_center - w * IMAGE_SIZE / 2
                y1 = y_center - h * IMAGE_SIZE / 2
                x2 = x_center + w * IMAGE_SIZE / 2
                y2 = y_center + h * IMAGE_SIZE / 2
                gt_boxes.append([x1, y1, x2, y2])
                # Get the class index from the one-hot encoded class probabilities
                class_probs = target[5:, i, j]
                gt_labels.append(torch.argmax(class_probs).item())
    if gt_boxes:
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
    else:
        gt_boxes = torch.empty((0, 4), dtype=torch.float32)
    return gt_boxes, np.array(gt_labels)


def evaluate_map(model, test_loader, iou_thresh, conf_thresh):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        for images, targets in progress_bar:
            images = images.to(device)
            predictions_batch = model.detect(
                images, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
            # Process each image in the batch
            for target, detections in zip(targets, predictions_batch):
                # Convert the ground truth target (from your collate_fn) to proper boxes and labels
                gt_boxes, gt_labels = decode_ground_truth(target)
                # Skip if there are no ground truth boxes
                if gt_boxes.numel() == 0:
                    continue
                # If no detections, skip this image
                if len(detections) == 0:
                    continue
                # Convert detections (a list) to a tensor
                det_tensor = torch.tensor(
                    detections, device=device, dtype=torch.float32)
                # Extract predicted boxes, scores, and class IDs
                pred_boxes = det_tensor[:, :4]
                scores = det_tensor[:, 4]
                # Note: detections[5] is class_score and detections[6] is class_id
                pred_labels = det_tensor[:, 6].int()
                # Compute IoU between each predicted box and each ground truth box
                iou = calculate_iou(pred_boxes.unsqueeze(0),
                                    gt_boxes.unsqueeze(0))
                matches = match_predictions(iou, iou_thresh)
                tp, _, conf, pred_cls = update_metrics(
                    matches, scores.cpu().numpy(), pred_labels.cpu().numpy(), gt_labels)
                all_tp.append(tp)
                all_conf.append(conf)
                all_pred_cls.append(pred_cls)
                all_target_cls.extend(gt_labels)
    all_tp = np.concatenate(all_tp)
    all_conf = np.concatenate(all_conf)
    all_pred_cls = np.concatenate(all_pred_cls)
    all_target_cls = np.concatenate(all_target_cls)
    ap_per_class = calculate_ap_per_class(
        all_tp, all_conf, all_pred_cls, all_target_cls)
    mean_ap = np.mean(list(ap_per_class.values()))
    print(f"Mean Average Precision: {mean_ap:.4f}")
    for cls, ap in ap_per_class.items():
        print(f"AP for class {cls}: {ap:.4f}")
    return mean_ap


class YOLOLoss(nn.modules.loss._Loss):
    def __init__(self, lambda_coord=LAMBDA_COORD, lambda_noobj=LAMBDA_NOOB, lambda_obj=LAMBDA_OBJ, lambda_class=LAMBDA_CLASS):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class

    def forward(self, predictions, target):
        pred = predictions.view(-1, NUMBER_OF_BBOXES * 5 +
                                NUMBER_OF_CLASSES, GRID_SIZE, GRID_SIZE).permute(0, 2, 3, 1)
        targ = target.view(-1, NUMBER_OF_BBOXES * 5 + NUMBER_OF_CLASSES,
                           GRID_SIZE, GRID_SIZE).permute(0, 2, 3, 1)

        pred_boxes = pred[..., :NUMBER_OF_BBOXES *
                          5].contiguous().view(-1, GRID_SIZE, GRID_SIZE, NUMBER_OF_BBOXES, 5)
        pred_classes = pred[..., NUMBER_OF_BBOXES * 5:]
        targ_boxes = targ[..., :NUMBER_OF_BBOXES *
                          5].contiguous().view(-1, GRID_SIZE, GRID_SIZE, NUMBER_OF_BBOXES, 5)
        targ_classes = targ[..., NUMBER_OF_BBOXES * 5:]

        obj_mask = targ_boxes[..., 4] > 0
        noobj_mask = targ_boxes[..., 4] == 0

        # Expand mask to cover class predictions
        obj_mask_for_classes = obj_mask.any(
            dim=-1).unsqueeze(-1).expand(-1, -1, -1, NUMBER_OF_CLASSES)
        masked_pred_classes = pred_classes[obj_mask_for_classes].view(
            -1, NUMBER_OF_CLASSES)
        masked_targ_classes = targ_classes[obj_mask_for_classes].view(
            -1, NUMBER_OF_CLASSES)

        box_loss = F.mse_loss(
            pred_boxes[obj_mask][..., :2], targ_boxes[obj_mask][..., :2], reduction='sum')
        box_loss += F.mse_loss(torch.sign(pred_boxes[obj_mask][..., 2:4]) * torch.sqrt(torch.abs(pred_boxes[obj_mask][..., 2:4])),
                               torch.sqrt(targ_boxes[obj_mask][..., 2:4]), reduction='sum')

        ious = bbox_iou(pred_boxes[..., :4], targ_boxes[..., :4]).detach()
        pred_obj = pred_boxes[..., 4] * ious
        obj_loss = F.mse_loss(
            pred_obj[obj_mask], targ_boxes[obj_mask][..., 4], reduction='sum')
        noobj_loss = F.mse_loss(
            pred_obj[noobj_mask], targ_boxes[noobj_mask][..., 4], reduction='sum')
        class_loss = F.mse_loss(masked_pred_classes,
                                masked_targ_classes, reduction='sum')

        loss = self.lambda_coord * box_loss + self.lambda_obj * obj_loss + \
            self.lambda_noobj * noobj_loss + self.lambda_class * class_loss
        loss /= predictions.size(0)
        return loss


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float,
                        default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float,
                        default=0.0005, help="Weight decay")
    parser.add_argument("--momentum", type=float,
                        default=0.90, help="Momentum")
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data",
                        help="Directory where dataset is stored")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model",
                        help="Directory to save the trained model")
    parser.add_argument("--load-weights", action="store_true",
                        help="Flag to load existing weights")
    parser.add_argument("--force-train", action="store_true",
                        help="Flag to force training even if weights exist")
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()

    # Create a SageMaker Debugger hook using the configuration provided
    # by SageMaker (this reads the configuration from the environment).
    hook = None
    if accelerator.is_main_process:
        try:
            hook = smd.Hook.create_from_json_file()
        except Exception as e:
            print("Could not create SM Debugger hook, continuing without debugger:", e)

    # Load datasets and data loaders.
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir)

    # Check for existing weights.
    weights = None
    modelos = sorted(glob.glob(os.path.join(
        args.model_dir, 'model_*.pth')), key=os.path.getmtime)
    if len(modelos) > 0:
        weights = modelos[-1]

    # Initialize model and register it with the debugger hook.
    model = YOLO(num_classes=NUMBER_OF_CLASSES)
    if hook is not None:
        hook.register_module(model)
    if weights is not None:
        print(f'Loading weights from {weights}')
        state_dict = torch.load(weights)
        model.load_state_dict(state_dict)

    # Define loss function and register it with the hook.
    loss_function = YOLOLoss()
    if hook is not None:
        hook.register_loss(loss_function)

    # Prepare the model for training with acceleration.
    model, train_loader, val_loader, test_loader = accelerator.prepare(
        model, train_loader, val_loader, test_loader)

    # Train the model (passing the hook so that training metrics are captured)
    model = train_model(model, train_loader, val_loader, args.model_dir, accelerator,
                        num_epochs=args.epochs, loss_function=loss_function, hook=hook)
    model.eval()

    # Evaluate the model on the test dataset.
    print("Evaluating model on test dataset...")
    evaluate_map(model, test_loader, iou_thresh=0.5, conf_thresh=0.5)


if __name__ == '__main__':
    main()
