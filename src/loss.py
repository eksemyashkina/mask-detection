from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    N = boxes1.size(0)
    M = boxes2.size(0)
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    x1_1 = x1_1.unsqueeze(1).expand(N, M)
    y1_1 = y1_1.unsqueeze(1).expand(N, M)
    x2_1 = x2_1.unsqueeze(1).expand(N, M)
    y2_1 = y2_1.unsqueeze(1).expand(N, M)
    x1_2 = x1_2.unsqueeze(0).expand(N, M)
    y1_2 = y1_2.unsqueeze(0).expand(N, M)
    x2_2 = x2_2.unsqueeze(0).expand(N, M)
    y2_2 = y2_2.unsqueeze(0).expand(N, M)
    interX1 = torch.max(x1_1, x1_2)
    interY1 = torch.max(y1_1, y1_2)
    interX2 = torch.min(x2_1, x2_2)
    interY2 = torch.min(y2_1, y2_2)
    interW = (interX2 - interX1).clamp(min=0)
    interH = (interY2 - interY1).clamp(min=0)
    interArea = interW * interH
    area1 = (x2_1 - x1_1).clamp(min=0) * (y2_1 - y1_1).clamp(min=0)
    area2 = (x2_2 - x1_2).clamp(min=0) * (y2_2 - y1_2).clamp(min=0)
    union = area1 + area2 - interArea + 1e-16
    iou = interArea / union
    return iou


def box_giou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    xA = torch.max(boxes1[:, 0], boxes2[:, 0])
    yA = torch.max(boxes1[:, 1], boxes2[:, 1])
    xB = torch.min(boxes1[:, 2], boxes2[:, 2])
    yB = torch.min(boxes1[:, 3], boxes2[:, 3])
    interW = (xB - xA).clamp(min=0)
    interH = (yB - yA).clamp(min=0)
    interArea = interW * interH
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - interArea + 1e-16
    iou = interArea / union
    xC1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    yC1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    xC2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    yC2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    encloseW = (xC2 - xC1).clamp(min=0)
    encloseH = (yC2 - yC1).clamp(min=0)
    encloseArea = encloseW * encloseH + 1e-16
    giou = iou - (encloseArea - union) / encloseArea
    return giou


class YoloLoss(nn.Module):
    def __init__(self, class_counts: List[int], anchors_l: List[int] = [(128, 152), (182, 205), (103, 124)], anchors_m: List[int] = [(78, 88), (55, 59), (37, 42)], anchors_s: List[int] = [(26, 28), (17, 19), (10, 11)], image_size: Tuple[int] = (416, 416), num_classes: int = 3, ignore_thresh: float = 0.7, lambda_noobj: float = 5.0):
        super().__init__()
        self.anchors_l = anchors_l
        self.anchors_m = anchors_m
        self.anchors_s = anchors_s
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.lambda_noobj = lambda_noobj
        total = sum(class_counts)
        w_list = [total / (c + 1e-5) * (2.0 if c_id == 0 else (3.0 if c_id == 2 else 1.0)) for c_id, c in enumerate(class_counts)]
        self.class_weight = torch.tensor(w_list, dtype=torch.float32)
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_cls = nn.BCEWithLogitsLoss(weight=self.class_weight, reduction="none")

    def forward(self, outputs: Tuple[torch.Tensor], targets: Tuple[torch.Tensor]) -> torch.Tensor:
        out_l, out_m, out_s = outputs
        t_l, t_m, t_s = targets
        loss_l = self._loss_single_scale(out_l, t_l, self.anchors_l, scale_wh=(13, 13))
        loss_m = self._loss_single_scale(out_m, t_m, self.anchors_m, scale_wh=(26, 26))
        loss_s = self._loss_single_scale(out_s, t_s, self.anchors_s, scale_wh=(52, 52))
        return loss_l + loss_m + loss_s

    def _loss_single_scale(self, pred: torch.Tensor, target: torch.Tensor, anchors: List[Tuple[int]], scale_wh: Tuple[int]) -> torch.Tensor:
        device = pred.device
        B, _, H, W = pred.shape
        A = len(anchors)
        pred = pred.view(B, A, (5 + self.num_classes), H, W)
        pred = pred.permute(0, 3, 4, 1, 2).contiguous()
        pred_tx = pred[..., 0]
        pred_ty = pred[..., 1]
        pred_tw = pred[..., 2]
        pred_th = pred[..., 3]
        pred_obj = pred[..., 4]
        pred_cls = pred[..., 5:]
        tgt_tx  = target[..., 0]
        tgt_ty  = target[..., 1]
        tgt_tw  = target[..., 2]
        tgt_th  = target[..., 3]
        tgt_obj = target[..., 4]
        tgt_cls = target[..., 5:]
        obj_mask = (tgt_obj == 1)
        noobj_mask = (tgt_obj == 0)
        img_w, img_h = self.image_size
        stride_x = img_w / W
        stride_y = img_h / H
        grid_x = torch.arange(W, device=device).view(1, 1, W, 1).expand(1, H, W, 1)
        grid_y = torch.arange(H, device=device).view(1, H, 1, 1).expand(1, H, W, 1)
        anchors_t = torch.tensor(anchors, dtype=torch.float, device=device)
        anchor_w = anchors_t[:, 0].view(1, 1, 1, A)
        anchor_h = anchors_t[:, 1].view(1, 1, 1, A)
        pred_box_xc = (grid_x + torch.sigmoid(pred_tx)) * stride_x
        pred_box_yc = (grid_y + torch.sigmoid(pred_ty)) * stride_y
        pred_box_w  = torch.exp(pred_tw) * anchor_w
        pred_box_h  = torch.exp(pred_th) * anchor_h
        pred_x1 = pred_box_xc - pred_box_w / 2
        pred_y1 = pred_box_yc - pred_box_h / 2
        pred_x2 = pred_box_xc + pred_box_w / 2
        pred_y2 = pred_box_yc + pred_box_h / 2
        gt_box_xc = (grid_x + tgt_tx) * stride_x
        gt_box_yc = (grid_y + tgt_ty) * stride_y
        gt_box_w  = torch.exp(tgt_tw) * anchor_w
        gt_box_h  = torch.exp(tgt_th) * anchor_h
        gt_x1 = gt_box_xc - gt_box_w / 2
        gt_y1 = gt_box_yc - gt_box_h  /2
        gt_x2 = gt_box_xc + gt_box_w / 2
        gt_y2 = gt_box_yc + gt_box_h / 2
        with torch.no_grad():
            ignore_mask_buf = torch.zeros_like(tgt_obj, dtype=torch.bool)
            noobj_flat = noobj_mask.view(-1)
            obj_flat = obj_mask.view(-1)
            px1f = pred_x1.view(-1)
            py1f = pred_y1.view(-1)
            px2f = pred_x2.view(-1)
            py2f = pred_y2.view(-1)
            gx1f = gt_x1.view(-1)[obj_flat]
            gy1f = gt_y1.view(-1)[obj_flat]
            gx2f = gt_x2.view(-1)[obj_flat]
            gy2f = gt_y2.view(-1)[obj_flat]
            if noobj_flat.sum() > 0 and obj_flat.sum() > 0:
                noobj_idx = noobj_flat.nonzero(as_tuple=True)[0]
                noobj_boxes_xyxy = torch.stack([px1f[noobj_idx], py1f[noobj_idx], px2f[noobj_idx], py2f[noobj_idx]], dim=-1)
                obj_boxes_xyxy = torch.stack([gx1f, gy1f, gx2f, gy2f], dim=-1)
                ious = box_iou_xyxy(noobj_boxes_xyxy, obj_boxes_xyxy)
                best_iou, _ = ious.max(dim=1)
                ignore_flags = (best_iou > self.ignore_thresh)
                all_idx = noobj_idx[ignore_flags]
                ignore_mask_buf.view(-1)[all_idx] = True
            ignore_mask = ignore_mask_buf
        obj_loss = self.bce_obj(pred_obj[obj_mask], torch.ones_like(pred_obj[obj_mask]))
        obj_loss = obj_loss.mean() if obj_loss.numel() > 0 else torch.tensor(0., device=device)
        noobj_mask_final = (noobj_mask & (~ignore_mask))
        noobj_loss = self.bce_obj(pred_obj[noobj_mask_final], torch.zeros_like(pred_obj[noobj_mask_final]))
        noobj_loss = noobj_loss.mean() if noobj_loss.numel() > 0 else torch.tensor(0., device=device)
        objectness_loss = obj_loss + self.lambda_noobj * noobj_loss
        class_loss = torch.tensor(0., device=device, requires_grad=True)
        if obj_mask.sum() > 0:
            self.bce_cls.weight = self.class_weight.to(device)
            cls_pred = pred_cls[obj_mask].to(device)
            cls_gt = tgt_cls[obj_mask].to(device)
            c_loss = self.bce_cls(cls_pred, cls_gt)
            class_loss = c_loss.mean()
        giou_loss = torch.tensor(0., device=device, requires_grad=True)
        if obj_mask.sum() > 0:
            px1_ = pred_x1[obj_mask]
            py1_ = pred_y1[obj_mask]
            px2_ = pred_x2[obj_mask]
            py2_ = pred_y2[obj_mask]
            p_xyxy = torch.stack([px1_,py1_,px2_,py2_], dim=-1)
            gx1_ = gt_x1[obj_mask]
            gy1_ = gt_y1[obj_mask]
            gx2_ = gt_x2[obj_mask]
            gy2_ = gt_y2[obj_mask]
            g_xyxy = torch.stack([gx1_,gy1_,gx2_,gy2_], dim=-1)
            giou = box_giou_xyxy(p_xyxy, g_xyxy)
            giou_loss = (1. - giou).mean()
        total_loss = objectness_loss + class_loss + giou_loss
        if total_loss is None:
            pass
        return total_loss