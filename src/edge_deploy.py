# Module 7 (Final): AV Production Safety Gate, ByteTrack + Ego-Lane Intrusion & Shadow-Mode Demo
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from ultralytics import YOLO

from bdd_loader import load_bdd_edge_frames
from local_logger import PALETTE, _style_ax, build_html_dashboard, save_metrics

load_dotenv()

RESEARCH_DATA_ROOT = Path(
    os.getenv(
        "RESEARCH_DATA_ROOT",
        "/Users/sadeeqadeyemi/Documents/Computervision_Research/Research_Data",
    )
)

# Production AV Safety Release Thresholds
SAFETY_GATES = {
    "ego_lane_recall": 0.95,   # >= 95% VRU recall inside driving corridor
    "macro_f1": 0.85,          # >= 85% Macro-F1 across all 4 VRU classes
    "min_class_recall": 0.85,  # No single VRU class may fall below 85%
    "precision": 0.90,         # >= 90% precision to avoid phantom braking
    "max_latency_ms": 33.3,    # <= 33.3 ms (>= 30 FPS real-time budget)
}

# Roboflow Cloud rfdetr-seg-medium (Version 1, Eval ID uuO7F5Wk9VTkn7XL0svJ) verified metrics
ROBOFLOW_RFDETR_METRICS = {
    "model_id": "sadeeqs-workspace/av-edge-case-engine-1-rfdetr-seg-medium-t1",
    "val_map50": 0.657,
    "macro_f1": 0.687,
    "precision": 0.774,
    "recall": 0.620,
    "min_class_recall": ("rider", 0.524),
    "pedestrian_med_large_map50": 0.976,
    "latency_ms": 22.37,
    "fps": 44.7,
}


def check_safety_gate() -> dict:
    """Evaluate our best trained model (RF-DETR-Seg-Medium) against real-world AV release gates."""
    m = ROBOFLOW_RFDETR_METRICS
    worst_cls, worst_val = m["min_class_recall"]

    checks = {
        "latency_ms (<= 33.3 ms / 30 FPS)": (m["latency_ms"], SAFETY_GATES["max_latency_ms"], m["latency_ms"] <= SAFETY_GATES["max_latency_ms"]),
        "macro_f1 (4-class valid)": (m["macro_f1"], SAFETY_GATES["macro_f1"], m["macro_f1"] >= SAFETY_GATES["macro_f1"]),
        f"min_class_recall ({worst_cls})": (worst_val, SAFETY_GATES["min_class_recall"], worst_val >= SAFETY_GATES["min_class_recall"]),
        "precision (4-class valid)": (m["precision"], SAFETY_GATES["precision"], m["precision"] >= SAFETY_GATES["precision"]),
    }

    passed_all = all(ok for _, _, ok in checks.values())
    verdict = "GO — ACTIVE CONTROL APPROVED" if passed_all else "SHADOW MODE ONLY — BLOCKED FOR ACTIVE CONTROL"

    print(f"\n=== AV DEPLOYMENT SAFETY GATE ({m['model_id']}) ===")
    print(f"Verdict: {verdict}")
    for name, (actual, target, ok) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<36} actual={actual:.3f} | required={target}")
    return {
        "model_id": m["model_id"],
        "verdict": verdict,
        "passed_all": passed_all,
        "checks": {k: {"actual": v[0], "target": v[1], "pass": v[2]} for k, v in checks.items()},
    }


def make_ego_lane_polygon(w: int, h: int) -> np.ndarray:
    """Construct a resolution-adaptive trapezoidal ego-lane driving corridor."""
    return np.array(
        [
            [int(w * 0.37), int(h * 0.57)],
            [int(w * 0.63), int(h * 0.57)],
            [int(w * 0.85), h],
            [int(w * 0.15), h],
        ],
        dtype=np.int32,
    )


def run_kitti_video_demo(
    weights: Path,
    seq_id: str = "0016",
    imgsz: int = 960,
    max_frames: int = 160,
    conf: float = 0.28,
    out_video: Path = Path("runs/edge_demo.mp4"),
    shadow_dir: Path = Path("runs/shadow_mode_queue"),
) -> dict:
    """Run Student v3 + ByteTrack + Motion Trails + Ego-Lane Alerts on continuous KITTI video."""
    seq_dir = RESEARCH_DATA_ROOT / "data_tracking_image_2/training/image_02" / seq_id
    frame_paths = sorted(seq_dir.glob("*.png"))[:max_frames]
    if not frame_paths:
        raise FileNotFoundError(f"No KITTI video frames found in {seq_dir}")

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    ego_poly = make_ego_lane_polygon(w, h)

    model = YOLO(str(weights))
    tracker = sv.ByteTrack(frame_rate=10)
    ego_zone = sv.PolygonZone(polygon=ego_poly)

    mask_annotator = sv.MaskAnnotator(opacity=0.35)
    box_annotator = sv.BoxAnnotator(thickness=2)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=15)
    label_annotator = sv.LabelAnnotator(text_scale=0.45, text_thickness=1, text_padding=3)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=ego_zone, color=sv.Color.GREEN, thickness=2, display_in_zone_count=False
    )

    out_video.parent.mkdir(parents=True, exist_ok=True)
    shadow_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))

    latencies_ms: list[float] = []
    unique_tracks: set[int] = set()
    ego_alert_frames = 0
    shadow_harvested = 0
    sample_gallery: list[dict[str, str]] = []
    preview_dir = Path("runs/reports/edge_deploy/frames")
    preview_dir.mkdir(parents=True, exist_ok=True)

    for idx, fpath in enumerate(frame_paths):
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        t0 = time.perf_counter()
        res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        dets = tracker.update_with_detections(dets)
        in_ego = ego_zone.trigger(detections=dets)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)

        if dets.tracker_id is not None:
            unique_tracks.update(int(tid) for tid in dets.tracker_id)

        labels: list[str] = []
        frame_alert = False
        low_conf_ego = False
        for i in range(len(dets)):
            cls_name = str(model.names[int(dets.class_id[i])])
            tid = int(dets.tracker_id[i]) if dets.tracker_id is not None else -1
            score = float(dets.confidence[i])
            if in_ego[i]:
                frame_alert = True
                if score < 0.45:
                    low_conf_ego = True
                labels.append(f"#{tid} {cls_name} EGO! {score:.2f}")
            else:
                labels.append(f"#{tid} {cls_name} {score:.2f}")

        annotated = zone_annotator.annotate(scene=img.copy())
        if dets.mask is not None and len(dets) > 0:
            annotated = mask_annotator.annotate(scene=annotated, detections=dets)
        annotated = trace_annotator.annotate(scene=annotated, detections=dets)
        annotated = box_annotator.annotate(scene=annotated, detections=dets)
        annotated = label_annotator.annotate(scene=annotated, detections=dets, labels=labels)

        # HUD bar
        fps_now = 1000.0 / max(1.0, lat_ms)
        hud_color = (0, 0, 210) if frame_alert else (25, 35, 55)
        cv2.rectangle(annotated, (0, 0), (w, 34), hud_color, -1)
        hud_text = (
            f"ALERT: VRU IN EGO LANE — BRAKE STANDBY  |  KITTI-{seq_id} #{idx:03d}  |  {fps_now:.1f} FPS"
            if frame_alert
            else f"AV SHADOW MODE  |  KITTI-{seq_id} #{idx:03d}  |  Tracks: {len(dets)}  |  {fps_now:.1f} FPS"
        )
        cv2.putText(annotated, hud_text, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        if frame_alert:
            ego_alert_frames += 1
        if low_conf_ego and shadow_harvested < 20:
            cv2.imwrite(str(shadow_dir / f"shadow_{seq_id}_{idx:03d}.jpg"), img)
            shadow_harvested += 1

        if idx % 20 == 0 or (frame_alert and len(sample_gallery) < 6):
            snap = preview_dir / f"demo_{seq_id}_{idx:03d}.jpg"
            cv2.imwrite(str(snap), annotated)
            if len(sample_gallery) < 6:
                sample_gallery.append(
                    {
                        "caption": f"KITTI-{seq_id} Frame {idx:03d} | {'EGO ALERT' if frame_alert else 'CLEAR'} | {len(dets)} tracked VRUs",
                        "teacher_img": str(snap),
                    }
                )

        writer.write(annotated)

    writer.release()
    avg_ms = float(np.mean(latencies_ms[3:] or latencies_ms or [1.0]))
    return {
        "sequence": f"KITTI-{seq_id}",
        "resolution": f"{w}x{h}",
        "frames_processed": len(frame_paths),
        "unique_vru_tracks": len(unique_tracks),
        "ego_lane_alert_frames": ego_alert_frames,
        "shadow_frames_harvested": shadow_harvested,
        "edge_ms": round(avg_ms, 1),
        "edge_fps": round(1000.0 / max(0.1, avg_ms), 1),
        "video_path": str(out_video.resolve()),
        "gallery": sample_gallery,
    }


def plot_latency_comparison(out_path: Path, local_ms: float, rfdetr_ms: float = 22.37, sam3_ms: float = 4800.0) -> Path:
    """Log-scale latency bar chart: SAM3 Teacher vs Local Student v3 vs Roboflow RF-DETR-Seg."""
    labels = ["SAM3 Teacher\n(Offline Labeler)", "Local Student v3\n(M5 + ByteTrack)", "RF-DETR-Seg-Med\n(Roboflow Cloud GPU)"]
    values = [sam3_ms, local_ms, rfdetr_ms]
    colors = [PALETTE["amber"], PALETTE["blue"], PALETTE["green"]]

    fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=PALETTE["bg"])
    bars = ax.bar(labels, values, color=colors, width=0.48)
    ax.set_yscale("log")
    _style_ax(
        ax,
        f"AV Distillation Speedup: Up to {sam3_ms / max(1.0, rfdetr_ms):.0f}x Faster Than SAM3 Teacher",
        ylabel="Latency (ms/frame, log scale)",
    )
    ax.axhline(33.3, color=PALETTE["red"], linestyle="--", linewidth=1.4, label="30 FPS Real-Time Budget (33.3 ms)")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v * 1.22,
            f"{v:.1f} ms\n({1000.0 / v:.1f} FPS)",
            ha="center",
            va="bottom",
            color=PALETTE["text"],
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_ylim(1, sam3_ms * 3.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close(fig)
    return out_path


def run_bdd_reel_demo(
    weights: Path,
    imgsz: int = 960,
    max_frames: int = 60,
    conf: float = 0.25,
    out_video: Path = Path("runs/edge_demo_bdd.mp4"),
) -> dict:
    """Render an annotated test reel on unseen BDD100K hard night/rain/weather VRU frames."""
    eval_path = Path("runs/eval_split.json")
    all_hard = load_bdd_edge_frames(limit=10000, require_vru=True, hard_only=True)
    if eval_path.exists():
        eval_names = json.loads(eval_path.read_text(encoding="utf-8"))
        by_name = {f.name: f for f in all_hard}
        frames = [by_name[n] for n in eval_names if n in by_name][:max_frames]
    else:
        frames = all_hard[:max_frames]

    w, h = 1280, 720
    ego_poly = make_ego_lane_polygon(w, h)
    model = YOLO(str(weights))
    ego_zone = sv.PolygonZone(polygon=ego_poly)

    mask_annotator = sv.MaskAnnotator(opacity=0.35)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=ego_zone, color=sv.Color.GREEN, thickness=2, display_in_zone_count=False
    )

    out_video.parent.mkdir(parents=True, exist_ok=True)
    # 2.5 FPS so each independent night/rain dashcam frame is clearly readable
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 2.5, (w, h))

    latencies_ms: list[float] = []
    ego_alert_frames = 0
    total_detections = 0

    for idx, frame in enumerate(frames):
        img = cv2.imread(frame.image_path)
        if img is None:
            continue
        img = cv2.resize(img, (w, h))

        t0 = time.perf_counter()
        res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        in_ego = ego_zone.trigger(detections=dets)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)
        total_detections += len(dets)

        labels: list[str] = []
        frame_alert = False
        for i in range(len(dets)):
            cls_name = str(model.names[int(dets.class_id[i])])
            score = float(dets.confidence[i])
            if in_ego[i]:
                frame_alert = True
                labels.append(f"{cls_name} EGO! {score:.2f}")
            else:
                labels.append(f"{cls_name} {score:.2f}")

        annotated = zone_annotator.annotate(scene=img.copy())
        if dets.mask is not None and len(dets) > 0:
            annotated = mask_annotator.annotate(scene=annotated, detections=dets)
        annotated = box_annotator.annotate(scene=annotated, detections=dets)
        annotated = label_annotator.annotate(scene=annotated, detections=dets, labels=labels)

        fps_now = 1000.0 / max(1.0, lat_ms)
        hud_color = (0, 0, 210) if frame_alert else (25, 35, 55)
        cv2.rectangle(annotated, (0, 0), (w, 42), hud_color, -1)
        cond = f"{frame.timeofday}/{frame.weather}"
        hud_text = (
            f"ALERT: VRU IN EGO LANE  |  BDD100K UNSEEN ({cond})  |  #{idx + 1:02d}/{len(frames)}  |  {fps_now:.1f} FPS"
            if frame_alert
            else f"AV SHADOW MODE  |  BDD100K UNSEEN ({cond})  |  VRUs: {len(dets)}  |  #{idx + 1:02d}/{len(frames)}  |  {fps_now:.1f} FPS"
        )
        cv2.putText(annotated, hud_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if frame_alert:
            ego_alert_frames += 1
        writer.write(annotated)

    writer.release()
    avg_ms = float(np.mean(latencies_ms[3:] or latencies_ms or [1.0]))
    return {
        "sequence": "BDD100K-Unseen-Hard",
        "resolution": f"{w}x{h}",
        "frames_processed": len(frames),
        "total_vru_detections": total_detections,
        "ego_lane_alert_frames": ego_alert_frames,
        "edge_ms": round(avg_ms, 1),
        "edge_fps": round(1000.0 / max(0.1, avg_ms), 1),
        "video_path": str(out_video.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 7: AV Safety Gate & Continuous Video Demo")
    parser.add_argument("--source", choices=["kitti", "bdd"], default="kitti")
    parser.add_argument("--weights", type=Path, default=Path("runs/student_v3/exp/weights/best.pt"))
    parser.add_argument("--seq", type=str, default="0016")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--out-video", type=Path, default=None)
    args = parser.parse_args()

    report_dir = Path("runs/reports/edge_deploy")
    gate = check_safety_gate()
    if args.source == "bdd":
        out_vid = args.out_video or Path("runs/edge_demo_bdd.mp4")
        demo = run_bdd_reel_demo(args.weights, imgsz=args.imgsz, max_frames=args.frames, out_video=out_vid)
        print(
            f"\nBDD100K reel saved -> {demo['video_path']} "
            f"({demo['frames_processed']} unseen frames | {demo['ego_lane_alert_frames']} ego alerts | {demo['edge_fps']} FPS)"
        )
        return

    out_vid = args.out_video or Path(f"runs/edge_demo_kitti{args.seq}.mp4")
    demo = run_kitti_video_demo(args.weights, seq_id=args.seq, imgsz=args.imgsz, max_frames=args.frames, out_video=out_vid)
    chart = plot_latency_comparison(report_dir / "latency_speedup.png", demo["edge_ms"])

    save_metrics(report_dir, {"safety_gate": gate, "video_demo": {k: v for k, v in demo.items() if k != "gallery"}})
    html = build_html_dashboard(
        report_dir,
        title="AV Data Engine — Module 7: Deployment Safety Gate & Live Video Demo",
        metrics={
            "Deployment Gate": "GO" if gate["passed_all"] else "SHADOW MODE",
            "RF-DETR Latency": f"{ROBOFLOW_RFDETR_METRICS['latency_ms']} ms ({ROBOFLOW_RFDETR_METRICS['fps']} FPS)",
            "RF-DETR Val mAP@50": f"{ROBOFLOW_RFDETR_METRICS['val_map50'] * 100:.1f}%",
            "Local M5 + ByteTrack": f"{demo['edge_fps']} FPS ({demo['edge_ms']} ms)",
            "Ego-Lane Alerts": demo["ego_lane_alert_frames"],
            "Shadow Hard Frames": demo["shadow_frames_harvested"],
        },
        chart_paths=[chart],
        gallery_items=demo["gallery"],
    )
    print(f"\nVideo demo saved -> {demo['video_path']}")
    print(f"Dashboard saved  -> {html}")


if __name__ == "__main__":
    main()
