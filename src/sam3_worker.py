#GENERATES MASKS FROM MINED FAILURE CASES

import json
import os
import sys 
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch



os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def mask_to_polygons(mask_bin: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 40:
            continue
        eps = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append(approx.flatten().astype(float).tolist())
    return polys


def main() -> None:
    manifest_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    items = json.loads(manifest_path.read_text(encoding="utf-8"))

    results: dict[str, list[dict]] = (
        json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    )
    todo = [entry for entry in items if entry["name"] not in results]
    if not todo:
        print(f"All {len(items)} frames already labeled in {out_path}.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device=device, load_from_HF=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.35)
    prompts = ["pedestrian", "rider", "bicycle", "motorcycle"]

    for idx, entry in enumerate(todo, start=1):
        img_path = entry["image_path"]
        image = Image.open(img_path).convert("RGB")
        state = processor.set_image(image)

        preds: list[dict] = []
        for prompt in prompts:
            out = processor.set_text_prompt(state=state, prompt=prompt)
            masks, boxes, scores = out["masks"], out["boxes"], out["scores"]

            if len(masks) == 0:
                continue

            m_np = masks.detach().cpu().numpy()
            if m_np.ndim == 4:
                m_np = m_np[:, 0]
            b_np = boxes.detach().cpu().numpy()
            s_np = scores.detach().cpu().numpy()

            for i in range(len(m_np)):
                mask_bin = (m_np[i] > 0.5).astype(np.uint8)

                if mask_bin.shape != (image.height, image.width):
                    mask_bin = cv2.resize(mask_bin, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

                preds.append({
                    "category": prompt,
                    "score": round(float(s_np[i]), 3),
                    "xyxy": [round(float(v), 1) for v in b_np[i].tolist()],
                    "polygons": mask_to_polygons(mask_bin),
                })

        results[entry["name"]] = preds
        print(f"[{idx}/{len(todo)}] SAM3 labeled {entry['name']}: {len(preds)} VRU masks")
        if idx % 5 == 0:
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()









