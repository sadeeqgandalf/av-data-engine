#lOADS DATSET

import json 
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel





load_dotenv()


VRU_CLASSES = {"pedestrian", "person", "rider", "bicycle", "motorcycle"}
HARD_WEATHER = {"rainy", "foggy", "snowy"}
HARD_TIME = {"night", "dawn/dusk"}

class GTBox(BaseModel):
    category: str
    xyxy: list[float]
    occluded: bool = False



class BDDFrame(BaseModel):
    name: str
    image_path: str
    width: int
    height: int
    weather: str
    scene: str
    timeofday: str
    is_hard_condition: bool
    vru_count: int
    occluded_vru_count: int
    boxes: list[GTBox]


def load_bdd_edge_frames(
    limit: int = 200,
    require_vru: bool = True,
    hard_only: bool = True,
) -> list [BDDFrame]:
  """Load real BDD100K frames for local disk filtered by VRU & weather/time conditions."""

  root = Path(os.environ["BDD100K_ROOT"])
  labels_file = root / "labels" / "det_val_simplified.json"
  images_dir = root / "images" / "100k" / "val_data"


  raw = json.loads(labels_file.read_text(encoding="utf-8"))
  frames: list[BDDFrame] = []


  for item in raw:
    img_path = images_dir / item["name"]
    if not img_path.exists():
        continue

    weather = item.get("weather", {}).get("label", "unknown")
    scene = item.get("scene", {}).get("label", "unknown")
    timeofday = item.get("timeofday", {}).get("label", "unknown")
    is_hard = (weather in HARD_WEATHER) or (timeofday in HARD_TIME)
    
    gt_boxes: list[GTBox] = []
    vru_count = 0
    occ_vru = 0



    for lbl in item.get("labels", []):
        cat = lbl.get("category", "")
        x, y, w, h = lbl["bbox"]
        occ = bool(lbl.get("occluded", False))
        gt_boxes.append(GTBox(category=cat, xyxy=[x, y, x+w, y+h], occluded= occ))
        

        if cat in VRU_CLASSES:
            vru_count += 1
            if occ:
                occ_vru += 1

    if require_vru and vru_count == 0:
            continue

    if hard_only and not (is_hard or occ_vru > 0):
            continue


    frames.append(
            BDDFrame(
                name = item["name"],
                image_path = str(img_path),
                width = item["width"],
                height = item["height"],
                weather = weather,
                scene = scene,
                timeofday = timeofday,
                is_hard_condition = is_hard,
                vru_count = vru_count,
                occluded_vru_count = occ_vru,
                boxes = gt_boxes,

            )
        )


    if len(frames) >= limit:
            break


  return frames

if __name__ == "__main__":                                                                                                                               
    hard_frames = load_bdd_edge_frames(limit=20, require_vru=True, hard_only=True)                                                                       
    print(f"Loaded {len(hard_frames)} real BDD100K hard VRU frames.")                                                                                    
    for f in hard_frames[:5]:                                                                                                                            
        print(f" - {f.name}: {f.timeofday}/{f.weather}/{f.scene} | VRUs={f.vru_count} (occluded={f.occluded_vru_count})")                                                                                                                      













