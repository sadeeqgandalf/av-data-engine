##COMPARES SAM3 MASK VS GT

import json
import os
import subprocess
from pathlib import Path
import cv2
from dotenv import load_dotenv
import numpy as np
import supervision as sv
import wandb

from bdd_loader import VRU_CLASSES, load_bdd_edge_frames
from student_edge import mine_failures

load_dotenv()


def run_sam3_teacher(limit: int = 10) -> None:
    frames = load_bdd_edge_frames(limit = limit, require_vru= True, hard_only = True)
    hard_frames = mine_failures(frames, Path("runs/preview"))
    if not hard_frames:
        print("No hard frames mined.")
        return

    manifest = Path("runs/sam3_manifest.json")
    labels_out = Path("runs/sam3_labels.json")
    manifest.write_text(json.dumps([{"name": f.name, "image_path": f.image_path} for f in hard_frames], indent=2))
    sam3_py = os.environ["SAM3_PYTHON"]
    subprocess.run([sam3_py, "src/sam3_worker.py", str(manifest), str(labels_out)], check=True)
    teacher_preds = json.loads(labels_out.read_text())

    compare_dir = Path("runs/teacher_compare")
    compare_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project="av-data-engine", name="sam3-teacher-audit")

    total_gt, teacher_hits = 0, 0
    for idx, frame in enumerate(hard_frames):
        img = cv2.imread(frame.image_path)
        t_list = teacher_preds.get(frame.name, [])
        gt_xyxy = np.array([b.xyxy for b in frame.boxes if b.category in VRU_CLASSES], dtype=np.float32)
        total_gt += len(gt_xyxy)

        if len(gt_xyxy) > 0 and len(t_list) > 0:
            t_xyxy = np.array([p["xyxy"] for p in t_list], dtype=np.float32)
            ious = sv.box_iou_batch(gt_xyxy, t_xyxy)
            teacher_hits += int((ious.max(axis=1) >= 0.35).sum())

        for p in t_list:
            for poly in p["polygons"]:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 120), thickness=2)
            x1, y1, x2, y2 = [int(v) for v in p["xyxy"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 120), 2)

        cv2.imwrite(str(compare_dir / f"teacher_{idx:03d}_{frame.name}"), img)
        wandb.log({
            "teacher/sam3_overlay": wandb.Image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption=f"{frame.name} | SAM3 VRUs={len(t_list)}",
            ),
        })

    teacher_recall = teacher_hits / max(1, total_gt)
    wandb.log({"eval/sam3_teacher_vru_recall": teacher_recall})
    wandb.finish()
    print(f"SAM3 Teacher VRU Recall on hard frames: {teacher_recall:.3f} -> {labels_out}")


if __name__ == "__main__":
    run_sam3_teacher(limit = 10)




