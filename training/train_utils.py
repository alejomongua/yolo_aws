# train_utils.py
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from train_config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM


def train_model(model, train_loader, validation_loader, num_epochs=NUM_EPOCHS, loss_function=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device '{device}'.")
    if loss_function is None:
        raise ValueError("A loss function must be provided.")
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    model.to(device)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for g in optimizer.param_groups:
            if epoch > 0 and epoch <= 5:
                g['lr'] = 0.00001 + (0.00009/5) * epoch
            elif epoch <= 80 and epoch > 5:
                g['lr'] = 0.0001
            elif epoch <= 110 and epoch > 80:
                g['lr'] = 0.00001
            elif epoch > 110:
                g['lr'] = 0.000001
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(train_loss=total_loss/len(train_loader))
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_val_loss = validate(
            model, validation_loader, loss_function, device)
        val_losses.append(avg_val_loss)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), f'model_{int(time.time())}.pth')
    return model


def validate(model, validation_loader, loss_function, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(validation_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            progress_bar.set_postfix(
                val_loss=total_loss/len(validation_loader))
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


def evaluate_map(model, test_loader, iou_thresh, conf_thresh):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        for images, _ in progress_bar:
            images = images.to(device)
            predictions = model.detect(
                images, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
            for _, detections in enumerate(predictions):
                gt_boxes = detections[:4, :, :].contiguous().view(-1, 4)
                gt_labels = torch.nonzero(
                    detections[4, :, :]).squeeze(1).cpu().numpy()
                if detections is None:
                    continue
                pred_boxes = detections[:, :4]
                scores = detections[:, 4]
                pred_labels = detections[:, 5]
                iou = calculate_iou(pred_boxes, gt_boxes)
                matches = match_predictions(iou, iou_thresh)
                matches = match_predictions(iou, iou_thresh)
                tp, _, conf, pred_cls = update_metrics(
                    matches, scores, pred_labels, gt_labels)
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
