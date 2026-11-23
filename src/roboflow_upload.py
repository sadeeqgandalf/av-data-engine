# Module 4: Export SAM3 pseudo-labels to COCO + split dataset ZIP and upload to Roboflow
import argparse
import json
import os
import random
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from roboflow import Roboflow

load_dotenv()

CLASS_MAP = {
    "pedestrian": 1,
    "rider": 2,
    "bicycle": 3,
    "motorcycle": 4,
}


def polygon_area(poly: list[float]) -> float:
    """Compute polygon area via shoelace formula."""
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def build_coco_dict(items: list[dict], preds: dict[str, list[dict]]) -> dict:
    """Build a COCO instance-segmentation dictionary for a list of manifest entries."""
    coco = {
        "info": {"description": "AV Data Engine - SAM3 Teacher Hard VRU Frames"},
        "categories": [
            {"id": cid, "name": name, "supercategory": "vru"}
            for name, cid in CLASS_MAP.items()
        ],
        "images": [],
        "annotations": [],
    }
    ann_id = 1

    for img_id, entry in enumerate(items, start=1):
        name = entry["name"]
        img_path = Path(entry["image_path"])
        if not img_path.exists():
            print(f"[warn] Missing image, skipping: {img_path}")
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        coco["images"].append(
            {"id": img_id, "file_name": name, "width": w, "height": h}
        )

        for p in preds.get(name, []):
            cid = CLASS_MAP.get(p["category"])
            polys = [poly for poly in p.get("polygons", []) if len(poly) >= 6]
            if not cid or not polys:
                continue

            x1, y1, x2, y2 = p["xyxy"]
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            area = sum(polygon_area(poly) for poly in polys) or (bw * bh)

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],
                    "area": round(area, 1),
                    "segmentation": polys,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return coco


def assign_splits(
    items: list[dict],
    preds: dict[str, list[dict]] | None = None,
    train_ratio: float = 0.80,
    valid_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Deterministically stratify manifest items into train / valid / test by rarest VRU class."""
    n = len(items)
    if n < 5:
        return {"train": list(items)}

    rng = random.Random(seed)
    # Bucket each image by its rarest VRU class so valid/test always contain rare classes
    rarity_order = ["motorcycle", "bicycle", "rider", "pedestrian"]
    buckets: dict[str, list[dict]] = {k: [] for k in rarity_order + ["other"]}

    for entry in items:
        cats = {p["category"] for p in (preds or {}).get(entry["name"], [])}
        chosen = next((c for c in rarity_order if c in cats), "other")
        buckets[chosen].append(entry)

    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for bucket_items in buckets.values():
        if not bucket_items:
            continue
        rng.shuffle(bucket_items)
        m = len(bucket_items)
        if m == 1:
            splits["train"].extend(bucket_items)
            continue
        n_train = max(1, int(round(m * train_ratio)))
        n_valid = max(1, int(round(m * valid_ratio))) if m >= 3 else 0
        if n_train + n_valid > m:
            n_valid = max(0, m - n_train)
        splits["train"].extend(bucket_items[:n_train])
        splits["valid"].extend(bucket_items[n_train : n_train + n_valid])
        splits["test"].extend(bucket_items[n_train + n_valid :])

    return {k: v for k, v in splits.items() if v}


def export_coco_and_zip(
    manifest_path: Path,
    labels_path: Path,
    out_coco: Path,
    out_zip: Path,
    auto_split: bool = True,
) -> tuple[Path, Path]:
    """Export full COCO JSON and a Roboflow-ready train/valid/test dataset ZIP."""
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    preds = json.loads(labels_path.read_text(encoding="utf-8"))

    full_coco = build_coco_dict(items, preds)
    out_coco.parent.mkdir(parents=True, exist_ok=True)
    out_coco.write_text(json.dumps(full_coco, indent=2), encoding="utf-8")
    print(
        f"Exported {len(full_coco['images'])} images, "
        f"{len(full_coco['annotations'])} SAM3 annotations -> {out_coco}"
    )

    splits = assign_splits(items, preds=preds) if auto_split else {"train": items}
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for split_name, split_items in splits.items():
            split_coco = build_coco_dict(split_items, preds)
            zf.writestr(
                f"{split_name}/_annotations.coco.json",
                json.dumps(split_coco, indent=2),
            )
            for entry in split_items:
                img_path = Path(entry["image_path"])
                if img_path.exists():
                    zf.write(img_path, arcname=f"{split_name}/{entry['name']}")

    split_summary = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Packaged ZIP ({size_mb:.2f} MB | {split_summary}) -> {out_zip}")
    return out_coco, out_zip


def upload_to_roboflow(
    manifest_path: Path,
    coco_path: Path,
    split: str = "train",
    batch_name: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Upload frames + SAM3 COCO annotations to Roboflow via Python SDK."""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace_slug = os.getenv("ROBOFLOW_WORKSPACE")
    project_slug = os.getenv("ROBOFLOW_PROJECT")

    if (
        not api_key
        or api_key == "your_roboflow_api_key_here"
        or not workspace_slug
        or workspace_slug == "your_workspace_slug"
        or not project_slug
    ):
        print(
            "Skipping SDK upload: set ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, and "
            "ROBOFLOW_PROJECT in .env (or upload runs/sam3_dataset.zip via Roboflow MCP)."
        )
        return

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_slug).project(project_slug)
    items = json.loads(manifest_path.read_text(encoding="utf-8"))

    batch = batch_name or f"sam3-hard-vru-{datetime.now().strftime('%Y%m%d-%H%M')}"
    tag_list = tags or ["sam3-teacher", "bdd100k-hard-vru"]

    uploaded = 0
    for entry in items:
        img_path = Path(entry["image_path"])
        if not img_path.exists():
            continue
        project.upload(
            image_path=str(img_path),
            annotation_path=str(coco_path),
            split=split,
            batch_name=batch,
            tag_names=tag_list,
            num_retry_uploads=2,
        )
        uploaded += 1
        print(f"Uploaded {entry['name']} -> {workspace_slug}/{project_slug} [{split}]")

    print(f"Upload complete: {uploaded}/{len(items)} frames sent to batch '{batch}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module 4: Export & upload SAM3 labels to Roboflow"
    )
    parser.add_argument("--manifest", type=Path, default=Path("runs/sam3_manifest.json"))
    parser.add_argument("--labels", type=Path, default=Path("runs/sam3_labels.json"))
    parser.add_argument("--coco-out", type=Path, default=Path("runs/sam3_coco.json"))
    parser.add_argument("--zip-out", type=Path, default=Path("runs/sam3_dataset.zip"))
    parser.add_argument("--no-auto-split", action="store_true")
    parser.add_argument("--skip-sdk-upload", action="store_true")
    args = parser.parse_args()

    coco_file, _ = export_coco_and_zip(
        args.manifest,
        args.labels,
        args.coco_out,
        args.zip_out,
        auto_split=not args.no_auto_split,
    )
    if not args.skip_sdk_upload:
        upload_to_roboflow(args.manifest, coco_file)


if __name__ == "__main__":
    main()
