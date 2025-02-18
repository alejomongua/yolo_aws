# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_config import (
    NUMBER_OF_BBOXES,
    NUMBER_OF_CLASSES,
    GRID_SIZE,
    LAMBDA_COORD,
    LAMBDA_NOOB,
    LAMBDA_OBJ,
    LAMBDA_CLASS,
)


class YOLOLoss(nn.Module):
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

        # Import bbox_iou from model.py for IOU computation
        from model import bbox_iou
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
