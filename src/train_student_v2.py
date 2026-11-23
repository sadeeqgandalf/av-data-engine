import argparse
import json
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from ultralytics import YOLO


from bdd_loader import VRU_CLASSES, load_bdd_edge_frames
from local_logger import PALETTE, _style_ax, build_html_dashboard, save_metrics



def prepare_yolo_seg_dataset(zip_path: Path, out_dir:Path) -> Path:
    """Convert Roboflow-style COCO polygon zip into Ultralytics YOLO-Seg Layout."""
    class_names: list[str] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for split in ["train", "valid", "test"]:
            coco_member = f" {split} /_annotations.json"
            if coco_member not in zf.namelist():
                continue

            img_dir = out_dir / "images" / split
            lbl_dir = out-dir / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir,mkdir(parents=True, exist_ok=True)



            coco = json.loads(zf.read(coco_member))
            cats = sorted(coco["categories"], key=lambda c: c["id"])
            cats_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
            if not class_names:
                class_names = [c["name"] for c in cats]

            anns_by_img: dict[int, list[dict]] = {}
            for ann in coco["annotations"]:
                anns_by_img.setdefault(ann["image_id"], []).append(ann)





