# Module 8: Train Student v4 (YOLO11s-Seg @ 960px on Fused Dataset) & 4-Way AV Benchmark
import argparse
import json
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from bdd_loader import BDDFrame, load_bdd_edge_frames
from local_logger import PALETTE, _style_ax, build_html_dashboard, save_metrics
from student_edge import EGO_LANE_POLYGON

TARGET_CLASSES = ["pedestrian", "rider", "bicycle", "motorcycle"]
SMALL_AREA_PX = 48 * 48


def prepare_yolo_seg_from_zip(zip_path: Path, out_dir: Path) -> Path:
    """Convert fused COCO polygon ZIP into Ultralytics YOLO-Seg directory layout."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Fused dataset ZIP not found: {zip_path}")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    class_names: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for split in ("train", "valid"):
            coco_member = f"{split}/_annotations.coco.json"
            if coco_member not in zf.namelist():
                continue

            img_dir = out_dir / "images" / split
            lbl_dir = out_dir / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            coco = json.loads(zf.read(coco_member))
            cats = sorted(coco["categories"], key=lambda c: c["id"])
            cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
            if not class_names:
                class_names = [c["name"] for c in cats]

            anns_by_img: dict[int, list[dict]] = defaultdict(list)
            for ann in coco["annotations"]:
                anns_by_img[ann["image_id"]].append(ann)

            for im in coco["images"]:
                fname, w, h = im["file_name"], im["width"], im["height"]
                (img_dir / fname).write_bytes(zf.read(f"{split}/{fname}"))

                lines: list[str] = []
                for ann in anns_by_img.get(im["id"], []):
                    cls_idx = cat_to_idx[ann["category_id"]]
                    for poly in ann.get("segmentation", []):
                        if len(poly) < 6:
                            continue
                        pts = [
                            f"{np.clip(poly[i] / w, 0.0, 1.0):.6f} {np.clip(poly[i + 1] / h, 0.0, 1.0):.6f}"
                            for i in range(0, len(poly), 2)
                        ]
                        lines.append(f"{cls_idx} " + " ".join(pts))

                (lbl_dir / f"{Path(fname).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    yaml_path = out_dir / "data.yaml"
    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/valid\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )
    print(f"Fused YOLO-Seg dataset ready -> {yaml_path}")
    return yaml_path


def _in_ego_lane(xyxy: list[float]) -> bool:
    cx, cy = 0.5 * (xyxy[0] + xyxy[2]), xyxy[3]
    return cv2.pointPolygonTest(EGO_LANE_POLYGON.astype(np.float32), (float(cx), float(cy)), False) >= 0


def _compute_ap(tp_flags: list[int], scores: list[float], total_gt: int) -> float:
    if total_gt == 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp_flags, dtype=np.float32)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    recalls = tp_cum / max(1, total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_student(
    model: YOLO, frames: list[BDDFrame], imgsz: int = 960, conf: float = 0.25, iou_thresh: float = 0.50
) -> dict:
    """Class-aware AV perception evaluation (mAP@50, Macro-F1, Ego-Lane, Small-VRU, FPS)."""
    class_gt, class_tp, class_fp = defaultdict(int), defaultdict(int), defaultdict(int)
    ap_tp, ap_scores = defaultdict(list), defaultdict(list)
    ego_gt = ego_hits = small_gt = small_hits = 0
    latencies_ms: list[float] = []

    for frame in frames:
        gt_boxes = [b for b in frame.boxes if b.category in TARGET_CLASSES]
        if not gt_boxes:
            continue

        gt_xyxy = np.array([b.xyxy for b in gt_boxes], dtype=np.float32)
        gt_cats = [b.category for b in gt_boxes]
        gt_ego = [_in_ego_lane(b.xyxy) for b in gt_boxes]
        gt_small = [(b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]) < SMALL_AREA_PX for b in gt_boxes]

        for c, is_e, is_s in zip(gt_cats, gt_ego, gt_small):
            class_gt[c] += 1
            ego_gt += int(is_e)
            small_gt += int(is_s)

        t0 = time.perf_counter()
        res = model.predict(frame.image_path, imgsz=imgsz, conf=conf, verbose=False)[0]
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        preds = sv.Detections.from_ultralytics(res)

        matched = np.zeros(len(gt_boxes), dtype=bool)
        if len(preds) > 0 and preds.confidence is not None and preds.class_id is not None:
            order = np.argsort(-preds.confidence)
            ious = sv.box_iou_batch(preds.xyxy, gt_xyxy)
            for p_idx in order:
                cname = str(model.names[int(preds.class_id[p_idx])])
                score = float(preds.confidence[p_idx])
                best_g, best_iou = -1, iou_thresh
                for g_idx, g_cat in enumerate(gt_cats):
                    if not matched[g_idx] and g_cat == cname and ious[p_idx, g_idx] >= best_iou:
                        best_iou = float(ious[p_idx, g_idx])
                        best_g = g_idx
                if best_g >= 0:
                    matched[best_g] = True
                    class_tp[cname] += 1
                    ap_tp[cname].append(1)
                    ap_scores[cname].append(score)
                elif cname in TARGET_CLASSES:
                    class_fp[cname] += 1
                    ap_tp[cname].append(0)
                    ap_scores[cname].append(score)

        ego_hits += sum(int(h and e) for h, e in zip(matched, gt_ego))
        small_hits += sum(int(h and s) for h, s in zip(matched, gt_small))

    per_class = {}
    for cls in TARGET_CLASSES:
        tp, fp, gt = class_tp[cls], class_fp[cls], class_gt[cls]
        p = tp / max(1, tp + fp)
        r = tp / max(1, gt)
        f1 = (2 * p * r) / max(1e-9, p + r)
        per_class[cls] = {
            "gt": gt,
            "tp": tp,
            "fp": fp,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "ap50": round(_compute_ap(ap_tp[cls], ap_scores[cls], gt), 4),
        }

    tot_tp, tot_fp, tot_gt = sum(class_tp.values()), sum(class_fp.values()), sum(class_gt.values())
    p_all, r_all = tot_tp / max(1, tot_tp + tot_fp), tot_tp / max(1, tot_gt)
    avg_ms = float(np.mean(latencies_ms[2:] or latencies_ms or [1.0]))

    return {
        "imgsz": imgsz,
        "total_gt": tot_gt,
        "precision": round(p_all, 4),
        "recall": round(r_all, 4),
        "f1": round((2 * p_all * r_all) / max(1e-9, p_all + r_all), 4),
        "macro_recall": round(float(np.mean([per_class[c]["recall"] for c in TARGET_CLASSES])), 4),
        "macro_f1": round(float(np.mean([per_class[c]["f1"] for c in TARGET_CLASSES])), 4),
        "map50": round(float(np.mean([per_class[c]["ap50"] for c in TARGET_CLASSES])), 4),
        "ego_lane_recall": round(ego_hits / max(1, ego_gt), 4),
        "small_vru_recall": round(small_hits / max(1, small_gt), 4),
        "latency_ms": round(avg_ms, 1),
        "fps": round(1000.0 / max(0.1, avg_ms), 1),
        "per_class": per_class,
    }


def grouped_bar(out_path: Path, title: str, cats: list[str], series: list[tuple[str, list[float], str]]) -> Path:
    x = np.arange(len(cats))
    width = 0.78 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(9.4, 4.8), facecolor=PALETTE["bg"])

    for idx, (label, vals, color) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.4,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                color=PALETTE["text"],
                fontsize=7.5,
                fontweight="bold",
            )

    _style_ax(ax, title, ylabel="Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 114)
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 8: Train Student v4 (YOLO11s-Seg on Fused Data)")
    parser.add_argument("--zip-path", type=Path, default=Path("runs/fused_dataset.zip"))
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path("runs/yolo_seg_v4")
    project_dir = Path("runs/student_v4").resolve()
    report_dir = Path("runs/reports/student_v4")
    best_weights = project_dir / "exp" / "weights" / "best.pt"

    yaml_path = prepare_yolo_seg_from_zip(args.zip_path, dataset_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if not args.skip_train or not best_weights.exists():
        print(f"Training Student v4 ({args.model}) on {device} | imgsz={args.imgsz}, epochs={args.epochs}...")
        trainer = YOLO(args.model)
        res = trainer.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            cos_lr=True,
            copy_paste=0.35,
            scale=0.5,
            project=str(project_dir),
            name="exp",
            exist_ok=True,
        )
        if getattr(res, "save_dir", None):
            best_weights = Path(res.save_dir) / "weights" / "best.pt"

    eval_names = json.loads(Path("runs/eval_split.json").read_text(encoding="utf-8"))
    by_name = {f.name: f for f in load_bdd_edge_frames(limit=10000, require_vru=True, hard_only=True)}
    eval_frames = [by_name[n] for n in eval_names if n in by_name]

    print(f"Evaluating Student v4 on {len(eval_frames)} fixed unseen hard VRU frames...")
    v4 = evaluate_student(YOLO(str(best_weights)), eval_frames, imgsz=args.imgsz)

    prev_path = Path("runs/reports/student_v3/metrics.json")
    models_dict = json.loads(prev_path.read_text())["models"] if prev_path.exists() else {}
    models_dict["Student v4 (11s, 960, Fused)"] = v4

    colors = [PALETTE["red"], PALETTE["amber"], PALETTE["blue"], PALETTE["green"]]
    entries = [(name, m, colors[i % len(colors)]) for i, (name, m) in enumerate(models_dict.items())]

    c1 = grouped_bar(
        report_dir / "v4_headline_metrics.png",
        "4-Way AV Flywheel Progression (100 Unseen Hard Frames, IoU>=0.50)",
        ["mAP@50", "Macro-F1", "Micro-F1", "Ego-Lane", "Small-VRU"],
        [
            (n, [m["map50"] * 100, m["macro_f1"] * 100, m["f1"] * 100, m["ego_lane_recall"] * 100, m["small_vru_recall"] * 100], c)
            for n, m, c in entries
        ],
    )
    c2 = grouped_bar(
        report_dir / "v4_per_class_recall.png",
        "Per-Class VRU Recall: Student v1 -> v2 -> v3 -> v4",
        TARGET_CLASSES,
        [(n, [m["per_class"][cls]["recall"] * 100 for cls in TARGET_CLASSES], c) for n, m, c in entries],
    )

    save_metrics(report_dir, {"eval_frames": len(eval_frames), "weights_path": str(best_weights), "models": models_dict})
    html = build_html_dashboard(
        report_dir,
        title="AV Data Engine — Student v4 (YOLO11s-Seg, Multi-Dataset Fused)",
        metrics={
            "mAP@50": f"{v4['map50'] * 100:.1f}%",
            "Macro-F1": f"{v4['macro_f1'] * 100:.1f}%",
            "Micro-F1": f"{v4['f1'] * 100:.1f}%",
            "Ego-Lane Recall": f"{v4['ego_lane_recall'] * 100:.1f}%",
            "Small-VRU Recall": f"{v4['small_vru_recall'] * 100:.1f}%",
            "Edge Speed (M5)": f"{v4['fps']} FPS ({v4['latency_ms']} ms)",
        },
        chart_paths=[c1, c2],
        gallery_items=[],
    )
    print(f"Done! Student v4: mAP@50={v4['map50']:.3f} | Macro-F1={v4['macro_f1']:.3f} | Ego={v4['ego_lane_recall']:.3f} | Report: {html}")


if __name__ == "__main__":
    main()
