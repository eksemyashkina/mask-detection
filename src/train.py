import math
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.ops as ops
import PIL
import numpy as np

from dataset import MaskDataset, collate_fn, ANCHORS
from utils import EMA
from models.yolov3 import YOLOv3
from loss import YoloLoss


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, eta_min: int = 0, last_epoch: int = -1) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch / max(1, self.warmup_steps)) for base_lr in self.base_lrs]
        else:
            current_step = self.last_epoch - self.warmup_steps
            cosine_steps = max(1, self.total_steps - self.warmup_steps)
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * current_step / cosine_steps)) for base_lr in self.base_lrs]


def draw_bounding_boxes(image: PIL.Image.Image, boxes: torch.Tensor, colors: Dict[int, int] = {0: (178, 34, 34), 1: (34, 139, 34), 2: (184, 134, 11)}, labels = {0: "without_mask", 1: "with_mask", 2: "weared_incorrect"}, show_conf = False) -> None:
    draw = PIL.ImageDraw.Draw(image)
    for box in boxes:
        xmin, ymin, xmax, ymax, class_id = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[-1])
        conf_text = ""
        if show_conf and box.shape[0] == 6:
            conf = float(box[4])
            conf_text = f" {conf:.2f}"
        color = colors.get(class_id, (255, 255, 255))
        label = labels.get(class_id, "Unknown") + conf_text
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        text_bbox = draw.textbbox((xmin, ymin), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([xmin, ymin - text_height - 2, xmin + text_width + 2, ymin], fill=color)
        draw.text((xmin + 1, ymin - text_height - 1), label, fill="white")


def create_combined_image(img: torch.Tensor, gt_batch: List[torch.Tensor], results: List[torch.Tensor], mean: List[float] = [0.485, 0.456, 0.406], std: List[float] = [0.229, 0.224, 0.225]):
    batch_size, _, height, width = img.shape
    combined_height = height * 2
    combined_width = width * batch_size
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    for i in range(batch_size):
        image = img[i].cpu().permute(1, 2, 0).numpy()
        image = (image * std + mean).clip(0, 1)
        image = (image * 255).astype(np.uint8)
        gt_image = PIL.Image.fromarray(image.copy())
        pred_image = PIL.Image.fromarray(image.copy())
        draw_bounding_boxes(gt_image, gt_batch[i])
        draw_bounding_boxes(pred_image, results[i], show_conf=True)
        combined_image[:height, i * width:(i + 1) * width, :] = np.array(gt_image)
        combined_image[height:, i * width:(i + 1) * width, :] = np.array(pred_image)
    return PIL.Image.fromarray(combined_image)


def decode_yolo_output_single(prediction: torch.Tensor, anchors: List[Tuple[int]], image_size: Tuple[int] = (416, 416), conf_threshold: float = 0.5, iou_threshold: float = 0.3, apply_nms: bool = True, num_classes: int = 3) -> List[torch.Tensor]:
    device = prediction.device
    B, _, H, W = prediction.shape
    A = len(anchors)
    prediction = prediction.view(B, A, 5 + num_classes, H, W)
    prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
    tx = prediction[..., 0]
    ty = prediction[..., 1]
    tw = prediction[..., 2]
    th = prediction[..., 3]
    obj = prediction[..., 4]
    class_scores = prediction[..., 5:]
    tx = tx.sigmoid()
    ty = ty.sigmoid()
    obj = obj.sigmoid()
    class_scores = class_scores.softmax(dim=-1)
    img_w, img_h = image_size
    cell_w = img_w / W
    cell_h = img_h / H
    grid_x = torch.arange(W, device=device).view(1, 1, W).expand(1, H, W)
    grid_y = torch.arange(H, device=device).view(1, H, 1).expand(1, H, W)
    anchors_tensor = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchor_w = anchors_tensor[:, 0].view(1, A, 1, 1)
    anchor_h = anchors_tensor[:, 1].view(1, A, 1, 1)
    x_center = (grid_x + tx) * cell_w
    y_center = (grid_y + ty) * cell_h
    w = torch.exp(tw) * anchor_w
    h = torch.exp(th) * anchor_h
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    max_class_probs, class_ids = class_scores.max(dim=-1)
    confidence = obj * max_class_probs
    outputs = []
    for b_i in range(B):
        box_xmin = xmin[b_i].view(-1)
        box_ymin = ymin[b_i].view(-1)
        box_xmax = xmax[b_i].view(-1)
        box_ymax = ymax[b_i].view(-1)
        conf = confidence[b_i].view(-1)
        cls_id = class_ids[b_i].view(-1).float()
        mask = (conf > conf_threshold)
        box_xmin = box_xmin[mask]
        box_ymin = box_ymin[mask]
        box_xmax = box_xmax[mask]
        box_ymax = box_ymax[mask]
        conf = conf[mask]
        cls_id = cls_id[mask]
        if mask.sum() == 0:
            outputs.append(torch.empty((0, 6), device=device))
            continue
        boxes = torch.stack([box_xmin, box_ymin, box_xmax, box_ymax], dim=-1)
        if apply_nms:
            keep = ops.nms(boxes, conf, iou_threshold)
            boxes = boxes[keep]
            conf = conf[keep]
            cls_id = cls_id[keep]
        out = torch.cat([boxes, conf.unsqueeze(-1), cls_id.unsqueeze(-1)], dim=-1)
        outputs.append(out)
    return outputs


def decode_predictions_3scales(out_l: torch.Tensor, out_m: torch.Tensor, out_s: torch.Tensor, anchors_l: List[Tuple[int]], anchors_m: List[Tuple[int, int]], anchors_s: List[Tuple[int, int]], image_size: Tuple[int, int] = (416, 416), conf_threshold: float = 0.5, iou_threshold: float = 0.45, num_classes: int = 3) -> List[torch.Tensor]:
    b_l = decode_yolo_output_single(out_l, anchors_l, image_size, conf_threshold, iou_threshold, apply_nms=False, num_classes=num_classes)
    b_m = decode_yolo_output_single(out_m, anchors_m, image_size, conf_threshold, iou_threshold, apply_nms=False, num_classes=num_classes)
    b_s = decode_yolo_output_single(out_s, anchors_s, image_size, conf_threshold, iou_threshold, apply_nms=False, num_classes=num_classes)
    results = []
    B = len(b_l)
    for i in range(B):
        boxes_all = torch.cat([b_l[i], b_m[i], b_s[i]], dim=0)
        if boxes_all.numel() == 0:
            results.append(boxes_all)
            continue
        xyxy = boxes_all[:, :4]
        scores = boxes_all[:, 4]
        keep = ops.nms(xyxy, scores, iou_threshold)
        final = boxes_all[keep]
        results.append(final)
    return results


def decode_target_single(target: torch.Tensor, anchors: List[Tuple[int]], image_size: Tuple[int] = (416, 416), obj_threshold: float = 0.5) -> List[torch.Tensor]:
    args = parse_args()
    target = target.to(args.device)
    B, S, _, A, _ = target.shape
    img_w, img_h = image_size
    cell_w = img_w / S
    cell_h = img_h / S
    anchors_tensor = torch.tensor(anchors, dtype=torch.float)
    tx = target[..., 0]
    ty = target[..., 1]
    tw = target[..., 2]
    th = target[..., 3]
    tobj = target[..., 4]
    tcls = target[..., 5:]
    results = []
    for b_i in range(B):
        bx_list = []
        tx_b = tx[b_i]
        ty_b = ty[b_i]
        tw_b = tw[b_i]
        th_b = th[b_i]
        tobj_b = tobj[b_i]
        tcls_b = tcls[b_i]
        for i in range(S):
            for j in range(S):
                for a_i in range(A):
                    if tobj_b[i,j,a_i] < obj_threshold:
                        continue
                    cls_one_hot = tcls_b[i, j, a_i]
                    cls_id = cls_one_hot.argmax().item()
                    x_center = (j + tx_b[i, j, a_i].item()) * cell_w
                    y_center = (i + ty_b[i, j, a_i].item()) * cell_h
                    anchor_w = anchors_tensor[a_i, 0]
                    anchor_h = anchors_tensor[a_i, 1]
                    box_w = torch.exp(tw_b[i, j, a_i]) * anchor_w
                    box_h = torch.exp(th_b[i, j, a_i]) * anchor_h
                    xmin = x_center - box_w / 2
                    ymin = y_center - box_h / 2
                    xmax = x_center + box_w / 2
                    ymax = y_center + box_h / 2
                    bx_list.append([xmin.item(), ymin.item(), xmax.item(), ymax.item(), cls_id])
        if len(bx_list) == 0:
            results.append(torch.empty((0, 5), dtype=torch.float32, device=args.device))
        else:
            results.append(torch.tensor(bx_list, dtype=torch.float32, device=args.device))
    return results


def decode_target_3scales(t_l: torch.Tensor, t_m: torch.Tensor, t_s: torch.Tensor, anchors_l: List[Tuple[int]], anchors_m: List[Tuple[int]], anchors_s: List[Tuple[int]], image_size: Tuple[int] = (416, 416), obj_threshold: float = 0.5) -> List[torch.Tensor]:
    dec_l = decode_target_single(t_l, anchors_l, image_size, obj_threshold)
    dec_m = decode_target_single(t_m, anchors_m, image_size, obj_threshold)
    dec_s = decode_target_single(t_s, anchors_s, image_size, obj_threshold)
    results = []
    B = len(dec_l)
    for i in range(B):
        boxes_l = dec_l[i]
        boxes_m = dec_m[i]
        boxes_s = dec_s[i]
        if boxes_l.numel() == 0 and boxes_m.numel() == 0 and boxes_s.numel() == 0:
            results.append(torch.empty((0, 5), dtype=torch.float32, device=boxes_l.device))
        else:
            all_ = torch.cat([boxes_l, boxes_m, boxes_s], dim=0)
            results.append(all_)
    return results


def iou_xyxy(box1: List[int | float], box2: List[int | float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0., x2 - x1)
    h = max(0., y2 - y1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap_per_class(boxes_pred: List[List[float]], boxes_gt: List[List[float]], iou_threshold: float = 0.45) -> float:
    boxes_pred = sorted(boxes_pred, key=lambda x: x[4], reverse=True)
    n_gt = len(boxes_gt)
    if n_gt == 0 and len(boxes_pred) == 0:
        return 1.0
    if n_gt == 0:
        return 0.0
    matched = [False] * n_gt
    tps = []
    fps = []
    for i, pred in enumerate(boxes_pred):
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(boxes_gt):
            if matched[j]:
                continue
            iou = iou_xyxy(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou > iou_threshold and best_j >= 0:
            tps.append(1)
            fps.append(0)
            matched[best_j] = True
        else:
            tps.append(0)
            fps.append(1)
    tps_cum = []
    fps_cum = []
    s_tp = 0
    s_fp = 0
    for i in range(len(tps)):
        s_tp += tps[i]
        s_fp += fps[i]
        tps_cum.append(s_tp)
        fps_cum.append(s_fp)
    precisions = []
    recalls = []
    for i in range(len(tps)):
        prec = tps_cum[i] / (tps_cum[i] + fps_cum[i]) if (tps_cum[i] + fps_cum[i]) > 0 else 0
        rec = tps_cum[i] / n_gt
        precisions.append(prec)
        recalls.append(rec)
    recalls = [0.0] + recalls + [1.0]
    precisions = [1.0] + precisions + [0.0]
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    ap = 0.0
    for i in range(len(precisions) - 1):
        ap += (recalls[i+1] - recalls[i]) * precisions[i+1]
    return ap


def compute_map(all_pred: List[float], all_gt: List[float], num_classes: int = 3, iou_threshold: float = 0.45) -> float:
    APs = []
    for c in range(num_classes):
        ap_c = compute_ap_per_class(all_pred[c], all_gt[c], iou_threshold)
        APs.append(ap_c)
    mAP = sum(APs) / len(APs) if len(APs) > 0 else 0.0
    return mAP


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the face mask detection dataset")
    parser.add_argument("--root", type=str, default="data/masks", help="Path to the data")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and testing")
    parser.add_argument("--logs-dir", type=str, default="yolo-logs", help="Path to save logs")
    parser.add_argument("--pin-memory", type=bool, default=True, help="Pin Memory for DataLoader")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for the optimizer")
    parser.add_argument("--save-frequency", type=int, default=4, help="Frequency of saving model weights")
    parser.add_argument("--max-norm", type=float, default=10.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--project-name", type=str, default="YOLOv3, mask detection", help="Wandb project name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the training on")
    parser.add_argument("--weights-path", type=str, default="weights/darknet53.pth", help="Path to the weights")
    parser.add_argument("--seed", type=int, default=42, help="Value of the seed")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["fp16", "bf16", "fp8", "no"], help="Value of the mixed precision")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Value of the gradient accumulation steps")
    parser.add_argument("--log-steps", type=int, default=13, help="Number of steps between logging training images and metrics")
    parser.add_argument("--num-warmup-steps", type=int, default=400, help="Number of steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    with accelerator.main_process_first():
        logs_dir = Path(args.logs_dir)
        logs_dir.mkdir(exist_ok=True)
        wandb.init(project=args.project_name, dir=logs_dir)
    train_dataset = MaskDataset(root=args.root, train=True)
    test_dataset = MaskDataset(root=args.root, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers, collate_fn=collate_fn)
    model = YOLOv3().to(accelerator.device)
    optimizer_class = getattr(torch.optim, args.optimizer)
    if args.weights_path:
        weights = torch.load(args.weights_path, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(weights)
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)
    criterion = YoloLoss(class_counts=train_dataset.class_counts)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_steps=args.num_warmup_steps//args.gradient_accumulation_steps, total_steps=args.num_epochs*len(train_loader)//args.gradient_accumulation_steps, eta_min=1e-7)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    best_map = 0.0
    train_loss_ema = EMA()
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc = f"Train epoch {epoch} / {args.num_epochs}")
        for images, (t_l, t_m, t_s) in pbar:
            images = images.to(accelerator.device)
            t_l = t_l.to(accelerator.device)
            t_m = t_m.to(accelerator.device)
            t_s = t_s.to(accelerator.device)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    out_l, out_m, out_s = model(images)
                    loss = criterion((out_l, out_m, out_s), (t_l, t_m, t_s))
                    accelerator.backward(loss)
                    grad_norm = None
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_norm).item()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({"loss": train_loss_ema(loss.item())})
                    log_data = {
                        "train/epoch": epoch,
                        "train/loss": loss.item(),
                        "train/lr": lr
                    }
                    if grad_norm is not None:
                        log_data["train/grad_norm"] = grad_norm
                    if accelerator.is_main_process:
                        wandb.log(log_data)
        accelerator.wait_for_everyone()
        model.eval()
        all_pred = [[] for _ in range(model.num_classes)]
        all_gt = [[] for _ in range(model.num_classes)]
        with torch.inference_mode():
            test_loss = 0.0
            pbar = tqdm(test_loader, desc=f"Test epoch {epoch} / {args.num_epochs}")
            for index, (images, (t_l, t_m, t_s)) in enumerate(pbar):
                images = images.to(accelerator.device)
                t_l = t_l.to(accelerator.device)
                t_m = t_m.to(accelerator.device)
                t_s = t_s.to(accelerator.device)
                out_l, out_m, out_s = model(images)
                loss = criterion((out_l, out_m, out_s), (t_l, t_m, t_s))
                test_loss += loss.item()
                results = decode_predictions_3scales(out_l, out_m, out_s, ANCHORS["large"], ANCHORS["medium"], ANCHORS["small"])
                gt_batch = decode_target_3scales(t_l, t_m, t_s, ANCHORS["large"], ANCHORS["medium"], ANCHORS["small"])
                if (index + 1) % args.log_steps == 0 and accelerator.is_main_process:
                    images_to_log = []
                    combined_image = create_combined_image(images, gt_batch, results)
                    images_to_log.append(wandb.Image(combined_image, caption=f"Combined Image (Test, Epoch {epoch})"))
                    wandb.log({"test_samples": images_to_log})
                for b_i in range(len(images)):
                    dets_b = results[b_i].detach().cpu().numpy()
                    gts_b = gt_batch[b_i].detach().cpu().numpy()
                    for db in dets_b:
                        c = int(db[5])
                        all_pred[c].append([db[0], db[1], db[2], db[3], db[4]])
                    for gb in gts_b:
                        c = int(gb[4])
                        all_gt[c].append([gb[0], gb[1], gb[2], gb[3]])
        test_loss /= len(test_loader)
        test_map = compute_map(all_pred, all_gt)
        accelerator.print(f"loss: {test_loss:.3f}, map: {test_map:.3f}")
        if accelerator.is_main_process:
            wandb.log({
                "epoch": epoch,
                "test/loss": test_loss,
                "test/mAP": test_map
            })
            if test_map > best_map:
                best_map = test_map
                accelerator.save(model.state_dict(), logs_dir / "checkpoint-best.pth")
            elif epoch % args.save_frequency == 0:
                accelerator.save(model.state_dict(), logs_dir / f"checkpoint-{epoch:09}.pth")
        accelerator.wait_for_everyone()
    accelerator.wait_for_everyone()
    wandb.finish()


if __name__ == "__main__":
    main()