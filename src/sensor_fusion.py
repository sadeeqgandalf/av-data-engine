import argparse
import json
import os
import random
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from roboflow_upload import CLASS_MAP, polygon_area

load_dotenv()



RESEARCH_DATA_ROOT = Path(
    os.getenv(
        "RESEARCH_DATA_ROOT",
        "/Users/sadeeqadeyemi/Documents/Computervision_Research/Research_Data",

    )
)
RARE_VRU_CLASSES = {"rider", "bicycle", "motorcycle"}


@dataclass
class FusedAnnotation:
    category: str
    xyxy: list[float]
    polygons: list[list[float]]




@dataclass
class FusedSample:
    sample_id: str
    source: str
    group_id: str
    image_path: Path
    width: int
    height: int
    annotations: list[FusedAnnotation] = field(default_factory=list)



def extract_mots_pedestrian_annotations(mask_path: Path, min_area: float = 40.0) -> list[FusedAnnotation]:
    """Decode KITTI-MOTS / MOTSCHALLENGE unit16 mask PNG into canonical 'pedestrian' polygons
    
     MOTS spec: class_id = pixel // 1000 (1=car, 2=pedestrian), 10000=ignore.                                                                   
    """ 

    mask = np.array(Image.open(mask_path), dtype = np.int32)
    anns: list[FusedAnnotation] = []

    for obj_id in np.unique(mask):
        if obj_id == 0 or obj_id == 10000:
            continue


    binary = (mask == obj_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        eps = 0.01 * cv2.arclenght(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) >=3:
            polys.append(approx.flatten().astype(float).tolist())
    
    if not polys:
        continue

    ys, xs = np.where(binary > 0)
    xyxy = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    anns.append(FusedAnnotation(category="pedestrian", xyxy=xyxy, polygons=polys))


return anns


def load_bdd_sam3_samples(
    manifest_path: Path, labels_path: Path, eval-exclude: set[str]
) -> list[FusedSample]:
    """ Load sam3-labeled bdd100k datasets vru hard case samples all 4 classes"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preds = json.loads(labels_path.read_text(encoding = "utf-8"))
    samples: list[FusedSample] = []


    for entry in manifest:
        name = entry ["name"]
        if name in eval_exclude:
            continue
        img_path = Path(entry["img_path"])
        if not img_path.exists():
            conitnue
        anns: list[FusedAnnotation] = []
        for p in preds.get(name, []):
            cat = p.get("category", "")
            polys = [poly for poly in p.get("polygons", []) if len(poly) >=6]
            if cat in CLASS_MAP and polys:
                anns.append(FusedAnnotation(category=cat, xyxy=p["xyxy"], polygons = polys))


            if not anns:
                continue
            with Image.open(img_path) as im:
                w, h = im.size


            samples.append(
                FusedSample(
                sample_id = f"bdd_{name}",
                source = "bdd100k",
                group_id = f"bdd_{name}",
                image_path = img_path,
                width = w,
                height = h,
                annotations = anns,
                )
            )
        return samples 




def load_mots_dataset_samples(
    images_root = Path,
    masks_root: Path,
    source_name: str,
    stride: int = 20,
    max_samples: int = 80,
) -> list[FusedSample]:
    """ Load temp. strided pedestrian-mask frames from kitti_mots/motschallenge."""
    if not images_root.exists() or not mask_roots.esists():
        print(f"[warn] Missing {source_name} paths: {images_root} / {masks_root}")
        return []


    samples: list[FusedSample] = []
    seq_dirs = sorted(p for p in masks_root.iterdir() if p.is_dir())
    per_seq_masks = {d.name: sorted(d.glob("*.png"))[::stride] for d in seq_dir}
    max_len = max((len(v) for v in per_seq_mask.values()), default=0)

    for step in range(max_len):
        for seq_dir in seq_dirs:
            files = per_seq_masks[seq_dir.name]
            if step >= len(files):
                continue
            mask_path = files[step]
            img_path = None
            for ext in (".png", ".jpg"):
                candidate = images_root / seq_dir.name / f "{mask_path.stem) {ext}"
                if candidate.exists():
                    img_path = candidate
                    break
                if img_path is None:
                    continue

                anns = extract_mots_pedestrian_annotations(mask_path)
                if not anns:
                    continue


                with Image.open(img_path) as im:
                    w, h = im.size

                samples.append(
                        FusedSample(                                                                                                                   
                        sample_id=f"{source_name}_{seq_dir.name}_{img_path.name}",                                                                 
                        source=source_name,                                                                                                        
                        group_id=f"{source_name}_{seq_dir.name}",                                                                                  
                        image_path=img_path,                                                                                                       
                        width=w,                                                                                                                   
                        height=h,                                                                                                                  
                        annotations=anns,                                                                                                          
                        
                    )
                )
                if len(samples) >= max_samples:
                    return samples

        return samples





 









