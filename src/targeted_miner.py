# Module 6: Targeted Long-Tail Active Learning Miner (Fix Class Imbalance + Stratified Eval Split)
import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from bdd_loader import BDDFrame, load_bdd_edge_frames
from roboflow_upload import export_coco_and_zip

load_dotenv()


def build_or_load_eval_split(
    all_frames: list[BDDFrame],
    exclude_names: set[str],
    eval_path: Path,
    per_class: int = 25,
) -> list[str]:
    """Create a fixed, stratified unseen benchmark split (motorcycle/bicycle/rider/pedestrian)."""
    if eval_path.exists():
        existing = json.loads(eval_path.read_text(encoding="utf-8"))
        print(f"Loaded fixed stratified eval split ({len(existing)} frames) from {eval_path}")
        return existing

    candidates = [f for f in all_frames if f.name not in exclude_names]
    selected: list[str] = []
    chosen_set: set[str] = set()

    for target_cls in ["motorcycle", "bicycle", "rider", "pedestrian"]:
        cls_frames = [
            f
            for f in candidates
            if f.name not in chosen_set
            and any(b.category == target_cls for b in f.boxes)
        ]
        # Prefer hard night/weather frames with multiple instances
        cls_frames.sort(
            key=lambda f: (
                int(f.is_hard_condition),
                sum(1 for b in f.boxes if b.category == target_cls),
                f.occluded_vru_count,
            ),
            reverse=True,
        )
        for f in cls_frames[:per_class]:
            selected.append(f.name)
            chosen_set.add(f.name)

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Created fixed stratified eval split ({len(selected)} unseen frames) -> {eval_path}")
    return selected


def mine_rare_class_frames(
    all_frames: list[BDDFrame],
    already_labeled: set[str],
    eval_names: set[str],
    n_motorcycle: int = 30,
    n_bicycle: int = 20,
    n_rider: int = 10,
) -> list[BDDFrame]:
    """Select high-value rare-class hard VRU frames strictly disjoint from eval & existing labels."""
    blocked = already_labeled | eval_names
    pool = [f for f in all_frames if f.name not in blocked]

    mined: list[BDDFrame] = []
    picked: set[str] = set()

    quotas = [
        ("motorcycle", n_motorcycle),
        ("bicycle", n_bicycle),
        ("rider", n_rider),
    ]
    for target_cls, quota in quotas:
        cls_pool = [
            f
            for f in pool
            if f.name not in picked and any(b.category == target_cls for b in f.boxes)
        ]
        cls_pool.sort(
            key=lambda f: (
                sum(1 for b in f.boxes if b.category == target_cls),
                int(f.is_hard_condition),
                f.vru_count,
            ),
            reverse=True,
        )
        take = cls_pool[:quota]
        mined.extend(take)
        picked.update(f.name for f in take)
        print(f"Targeted miner picked {len(take)} new '{target_cls}' hard frames")

    return mined


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 6: Targeted rare-class SAM3 mining")
    parser.add_argument("--motorcycles", type=int, default=30)
    parser.add_argument("--bicycles", type=int, default=20)
    parser.add_argument("--riders", type=int, default=10)
    parser.add_argument("--eval-per-class", type=int, default=25)
    args = parser.parse_args()

    manifest_path = Path("runs/sam3_manifest.json")
    labels_path = Path("runs/sam3_labels.json")
    eval_path = Path("runs/eval_split.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    already_names = {entry["name"] for entry in manifest}

    all_hard = load_bdd_edge_frames(limit=10000, require_vru=True, hard_only=True)
    eval_names = set(
        build_or_load_eval_split(
            all_hard,
            exclude_names=already_names,
            eval_path=eval_path,
            per_class=args.eval_per_class,
        )
    )

    new_frames = mine_rare_class_frames(
        all_hard,
        already_labeled=already_names,
        eval_names=eval_names,
        n_motorcycle=args.motorcycles,
        n_bicycle=args.bicycles,
        n_rider=args.riders,
    )

    if new_frames:
        for f in new_frames:
            manifest.append({"name": f.name, "image_path": f.image_path})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Updated manifest -> {len(manifest)} total frames ({len(new_frames)} newly mined)")

    sam3_py = os.environ["SAM3_PYTHON"]
    subprocess.run(
        [sam3_py, "src/sam3_worker.py", str(manifest_path), str(labels_path)],
        check=True,
    )

    preds = json.loads(labels_path.read_text(encoding="utf-8"))
    counts = Counter(p["category"] for plist in preds.values() for p in plist)
    print(f"Round 2 SAM3 mask distribution across {len(preds)} frames: {dict(counts)}")

    export_coco_and_zip(
        manifest_path=manifest_path,
        labels_path=labels_path,
        out_coco=Path("runs/sam3_coco.json"),
        out_zip=Path("runs/sam3_dataset.zip"),
        auto_split=True,
    )


if __name__ == "__main__":
    main()
