#MINES FAILURE CASES ON DATASET

import os
from pathlib import Path
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import numpy as np
import supervision as sv
import wandb


from bdd_loader import BDDFrame, VRU_CLASSES, load_bdd_edge_frames


load_dotenv()


EGO_LANE_POLYGON = np.array([
    [470, 420],
    [810, 420],
    [1140, 720],
    [140, 720],

],  dtype=np.int32)



def get_student_detections(image_path: str, client :InferenceHTTPClient | None) -> sv.Detections:
    """Run roboflow workflow on a baseline detector."""
    workflow_id = os.getenv("ROBOFLOW_WORKFLOW_ID")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")


    if client and workflow_id and workspace:
        result = client.run_workflow(
            workspace_name = workspace,
            workflow_id = workflow_id,
            images = {"image": image_path},

        )

        preds = result[0].get("predictions", {})
        return sv.Detections.from_inference(preds)

    from ultralytics import YOLO
    if not hasttr(get_student_detections, "_model"):
        get_student_detections._model = YOLO("yolo11n.pt")
    res = get_student_detections.model.predict(image_path, conf = 0.20, verbose=False)[0]



def mine_failures(frames: list[BDDFrame], out_dir: Path) -> list [BDDFrame]:
    """Run fast student + Ego-Lane and log failure overlays to disk and W&B."""
    out_dir.mkdir(parents = True, exist_ok = True)
    api_key = os.getenv("ROBOFLOW-API_KEY")
    client = InferenceHTTPClient(api_url = "https://detect.roboflow.com", api_key = api_key) if api_key else None

    zone = sv.PolygonZone(polygon = EGO_LANE_POLYGON)
    box_annotator = sv.BoxAnnotator(thickness =2)
    zone_annotator = sv.PolygonZoneAnnotator(zone = zone, color = sv.Color.GREEN, thickness=2)


    wandb.init(project="av-data-engine", name = "student-v1-bdd-mining")
    mined : list[BDDFRame] = []
    total_gt_vru = 0
    matched_gt_vru = 0




    for  idx, frame in enumerate(frames):
        img = cv2.imread(frame.image_path)
        preds = get_student_detections(frame.image_path,client)
        in_ego = zone.trigger(detections=preds)




        ##gt VRU boxes
        gt_xyxy = np.array([b.xyxy for b in frame.boxes if b.category in VRU_CLASSES], dtype = np.float32)
        total_gt_vru += len(gt_xyxy)

        frame_missed_vru = 0
        if len(gt_xyxy) >0:
            if len(preds) > 0:
                ious = sv.box_iou_batch(gt_xyxy, preds.xyxy)
                hits = (ious.max(axis=1) >= 0.35)
                matched_gt_vru += int(hits.sum())
                frame_missed_vru = int((~hits).sum())
            else:
                frame_missed_vru = len(gt_xyxy)


        low_conf_ego = bool(np.any((preds.confidence < 0.45) & in_ego)) if len(preds) > 0 else False
        is_hard_failure = (frame_missed_vru > 0) or low_conf_ego



        ##Annotate and save preview image for inspection
        annotated = box_annotator.annotate(scene = annotated, detection = preds)
        
        for box in gt_xyxy:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2) 


        if is_hard_failure:
            mined.append(frame)
            cv2.imwrite(str(out_dir / f" hard_{idx:03d}_{frame.name}"), annotated)
            wandb.log({
                "mining/hard_frame": wandb.Image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption = f"{frame.name} | missed_vru = {frame_missed_vru}"),
            })


    recall = matched_gt_vru / max(1, total_gt_vru)
    wandb.log({"eval/student_v1_vru_recall": recall, "mining/hard_frames-count": len(mined)})
    wandb.finish()
    print (f"Student v1 VRU Recall: {recall:.3f} | Mined {len(mined)}/{len(frames)} hard failure frames -> {out_dir}")
    return mined




if __name__ == "__main__":                                                                                                                               
        frames = load_bdd_edge_frames(limit=30, require_vru=True, hard_only=True)                                                                            
        mine_failures(frames, Path("runs/preview"))        

