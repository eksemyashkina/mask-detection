from typing import List, Tuple, Dict
from pathlib import Path
import PIL.Image
import numpy as np
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset
from bs4 import BeautifulSoup
from bs4.element import Tag


ANCHORS = {
    "small":  [(26, 28), (17, 19), (10, 11)],
    "medium": [(78, 88), (55, 59), (37, 42)],
    "large":  [(128, 152), (182, 205), (103, 124)]
}
GRID_SIZES = [13, 26, 52]
IMAGE_SIZE = (416, 416)
NUM_CLASSES = 3


def generate_box(obj: Tag) -> List[int]:
    xmin = int(obj.find("xmin").text) - 1
    ymin = int(obj.find("ymin").text) - 1
    xmax = int(obj.find("xmax").text) - 1
    ymax = int(obj.find("ymax").text) - 1
    if obj.find("name").text == "without_mask":
        class_id = 0
    elif obj.find("name").text == "with_mask":
        class_id = 1
    else:
        class_id = 2
    return [xmin, ymin, xmax, ymax, class_id]


def resize_boxes(box: List[int], scale: float, pad_x: int, pad_y: int) -> Tuple[int]:
    xmin, ymin, xmax, ymax, class_id = box
    xmin = int(xmin * scale + pad_x)
    ymin = int(ymin * scale + pad_y)
    xmax = int(xmax * scale + pad_x)
    ymax = int(ymax * scale + pad_y)
    return (xmin, ymin, xmax, ymax, class_id)


def resize_with_padding(image: PIL.Image.Image, target_size: Tuple[int] = IMAGE_SIZE, fill: Tuple[int] = (255, 255, 255)) -> Tuple[PIL.Image.Image, float, int]:
    target_w, target_h = target_size
    orig_w, orig_h = image.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    image_resized = image.resize((new_w, new_h), resample=PIL.Image.LANCZOS)
    new_image = PIL.Image.new("RGB", (target_w, target_h), color=fill)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    new_image.paste(image_resized, (pad_x, pad_y))
    return new_image, scale, pad_x, pad_y


def build_targets_3scale(bboxes: List[Tuple[int]], image_size: Tuple[int] = IMAGE_SIZE, anchors: Dict[str, List[Tuple[int]]] = ANCHORS, grid_sizes: List[int] = GRID_SIZES, num_classes: int = NUM_CLASSES) -> Tuple[torch.Tensor]:
    img_w, img_h = image_size
    t_large = torch.zeros((grid_sizes[0], grid_sizes[0], 3, 5 + num_classes), dtype=torch.float32)
    t_medium = torch.zeros((grid_sizes[1], grid_sizes[1], 3, 5 + num_classes), dtype=torch.float32)
    t_small = torch.zeros((grid_sizes[2], grid_sizes[2], 3, 5 + num_classes), dtype=torch.float32)
    all_anchors = anchors["large"] + anchors["medium"] + anchors["small"]
    for (xmin, ymin, xmax, ymax, cls_id) in bboxes:
        box_w = xmax - xmin
        box_h = ymax - ymin
        x_center = (xmax + xmin) / 2
        y_center = (ymax + ymin) / 2
        if box_w <= 0 or box_h <= 0:
            continue
        best_iou = 0
        best_idx = 0
        for i, (aw, ah) in enumerate(all_anchors):
            inter = min(box_w, aw) * min(box_h, ah)
            union = box_w * box_h + aw * ah - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx <= 2:
            s = grid_sizes[0]
            t = t_large
            local_anchor_id = best_idx
            anchor_w, anchor_h = anchors["large"][local_anchor_id]
        elif best_idx <= 5:
            s = grid_sizes[1]
            t = t_medium
            local_anchor_id = best_idx - 3
            anchor_w, anchor_h = anchors["medium"][local_anchor_id]
        else:
            s = grid_sizes[2]
            t = t_small
            local_anchor_id = best_idx - 6
            anchor_w, anchor_h = anchors["small"][local_anchor_id]
        cell_w = img_w / s
        cell_h = img_h / s
        gx = int(x_center // cell_w)
        gy = int(y_center // cell_h)
        tx = (x_center / cell_w) - gx
        ty = (y_center / cell_h) - gy
        tw = np.log((box_w / (anchor_w + 1e-16)) + 1e-16)
        th = np.log((box_h / (anchor_h + 1e-16)) + 1e-16)
        t[gy, gx, local_anchor_id, 0] = tx
        t[gy, gx, local_anchor_id, 1] = ty
        t[gy, gx, local_anchor_id, 2] = tw
        t[gy, gx, local_anchor_id, 3] = th
        t[gy, gx, local_anchor_id, 4] = 1.0
        t[gy, gx, local_anchor_id, 5 + cls_id] = 1.0
    return t_large, t_medium, t_small


class MaskDataset(Dataset):
    def __init__(self, root: str, train: bool = True, test_size: float = 0.25) -> None:
        super().__init__()
        self.class_counts = [0, 0, 0]
        self.root = root
        self.train = train
        all_imgs = sorted(list((Path(root) / "images").glob("*.png")))
        all_anns = sorted(list((Path(root) / "annotations").glob("*.xml")))
        n_test = int(len(all_imgs) * test_size)
        if train:
            self.images = all_imgs[n_test:]
            self.annots = all_anns[n_test:]
        else:
            self.images = all_imgs[:n_test]
            self.annots = all_anns[:n_test]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for ann in self.annots:
            with open(ann, "r") as f:
                data = f.read()
                soup = BeautifulSoup(data, "lxml")
                for obj in soup.find_all("object"):
                    cls = obj.find("name").text
                    self.class_counts[0 if cls == "without_mask" else 1 if cls == "with_mask" else 2] += 1

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        img_path = self.images[idx]
        ann_path = self.annots[idx]
        img = PIL.Image.open(img_path).convert("RGB")
        img_resized, scale, pad_x, pad_y = resize_with_padding(img)
        with open(ann_path, "r") as f:
            data = f.read()
            soup = BeautifulSoup(data, "lxml")
            objs = soup.find_all("object")
        resized_boxes = []
        for obj in objs:
            b = generate_box(obj)
            b2 = resize_boxes(b, scale, pad_x, pad_y)
            resized_boxes.append(b2)
        t_large, t_medium, t_small = build_targets_3scale(resized_boxes)
        img_tensor = self.transform(img_resized)
        return img_tensor, (t_large, t_medium, t_small)


def collate_fn(batch: List[Tuple[torch.Tensor, Tuple[torch.Tensor]]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
    imgs, t_l, t_m, t_s = [], [], [], []
    for (img, (tl, tm, ts)) in batch:
        imgs.append(img)
        t_l.append(tl)
        t_m.append(tm)
        t_s.append(ts)
    imgs = torch.stack(imgs, dim=0)
    t_l = torch.stack(t_l, dim=0)
    t_m = torch.stack(t_m, dim=0)
    t_s = torch.stack(t_s, dim=0)
    return imgs, (t_l, t_m, t_s)
