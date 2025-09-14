#!/usr/bin/env python3
"""
merge_and_label_coco.py

Merge multiple COCO datasets with inconsistent category names into a single, clean COCO dataset.
Additionally, generate image-level labels for two classification tasks:
 - damage (битый/небитый)
 - cleanliness (грязный/чистый) + optional levels based on dirt area ratio

USAGE:
  python merge_and_label_coco.py --root ./data --out ./data/_merged_coco \
      --splits train valid test \
      --dirty-thresholds 0.05 0.15

Notes:
- Designed for Roboflow-style structure where each dataset has train/valid/test subfolders
  with an "_annotations.coco.json".
- Images are copied into the merged dataset to avoid path issues.
- Category normalization handles synonyms and typos (dent/dunt, scratch/scracth, etc.).
- "car" boxes are kept (if present) to estimate dirt ratio relative to car area; otherwise
  the full image area is used as denominator.

Output structure:
  <out>/
    train/
      images/...
      _annotations.coco.json
    valid/
      images/...
      _annotations.coco.json
    test/
      images/...
      _annotations.coco.json
    multitask_labels_train.csv
    multitask_labels_valid.csv
    multitask_labels_test.csv
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv

def norm(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalnum())

def canonical_category(name: str) -> Optional[str]:
    """
    Map raw category name to a canonical one from:
      'car', 'dent', 'scratch', 'rust', 'dirt'

    Returns None if the category should be ignored.
    """
    n = norm(name)
    if not n:
        return None

    # car / vehicle
    if n in {"car", "carj87j"} or n.startswith("car"):
        return "car"

    # dirt / mud / dust - including bird poop and leaves as dirt
    if ("dirt" in n or "birdpoop" in n or "leaf" in n or 
        "mud" in n or "dust" in n):
        return "dirt"

    # rust
    if "rust" in n:
        return "rust"

    # dent/dunt - including specific car part dents
    if ("dent" in n or "dunt" in n or 
        # specific dent types
        "bonnentdent" in n or "doorouterdent" in n or "fenderdent" in n or
        "frontbumperdent" in n or "mediumbodypaneldent" in n or "pillardent" in n or
        "quaterpaneldent" in n or "rearbumperdent" in n or "roofdent" in n or
        "runningboarddent" in n or "majorrearbumperdent" in n):
        return "dent"

    # scratch (incl. scracth typos or car-scratch variants)
    if "scratch" in n or "scracth" in n or "carscratch" in n:
        return "scratch"

    # damage to specific car parts - treat as general damage (dent category for now)
    if (n in {"frontwindscreendamage", "headlightdamage", "rearwindscreendamage", 
              "sidemirrordamage", "signlightdamage", "taillightdamage"} or
        "damage" in n):
        return "dent"  # treating general damage as dent for now

    # General damage category - map to dent as generic damage
    if n == "damagedetection":
        return "dent"  # treat general damage as dent category

    # Known dataset-level supercategory name; ignore if it appears as a category by mistake.
    if n in {"rustscracthdunt"}:
        return None

    return None

CANONICAL_CATS = ["car", "dent", "scratch", "rust", "dirt"]
CAT_ID = {name: i+1 for i, name in enumerate(CANONICAL_CATS)}  # COCO ids start at 1

def load_coco(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_image(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)

def merge_split(split: str, dataset_dirs: List[Path], out_root: Path) -> Tuple[Dict, Dict[int, Dict]]:
    """
    Merge a specific split (train/valid/test) over all dataset_dirs.
    Returns:
        merged_coco: COCO dict
        img_idx: mapping new_image_id -> {"path": <rel_path>, "width": W, "height": H}
    """
    out_split_dir = out_root / split
    out_images_dir = out_split_dir / "images"
    ensure_dir(out_images_dir)

    merged = {
        "images": [],
        "annotations": [],
        "categories": [{"id": CAT_ID[name], "name": name, "supercategory": "none"} for name in CANONICAL_CATS],
    }

    next_img_id = 1
    next_ann_id = 1
    img_idx: Dict[int, Dict] = {}

    for d in dataset_dirs:
        ann_path = d / split / "_annotations.coco.json"
        if not ann_path.exists():
            print(f"[WARN] Missing {ann_path}, skipping.")
            continue

        ds = load_coco(ann_path)

        # Map raw category id -> canonical id (or None if ignored)
        raw_id_to_canonical: Dict[int, Optional[int]] = {}
        for c in ds.get("categories", []):
            can = canonical_category(c.get("name", ""))
            raw_id_to_canonical[c["id"]] = CAT_ID[can] if can in CAT_ID else None

        # Build image_id mapping and copy images
        raw_imgid_to_new: Dict[int, int] = {}
        base_images_dir = d / split  # Roboflow usually stores images right next to annotations

        for img in ds.get("images", []):
            raw_img_id = img["id"]
            file_name = img["file_name"]
            src = base_images_dir / file_name

            # make unique target name to avoid collisions
            # prefix with dataset folder name
            unique_name = f"{d.name}__{file_name}"
            dst = out_images_dir / unique_name

            if not src.exists():
                # Try common alternative layouts (e.g., images/ subfolder)
                alt = base_images_dir / "images" / file_name
                if alt.exists():
                    src = alt
                else:
                    print(f"[WARN] Image not found: {src}. Skipping this image+annotations.")
                    continue

            copy_image(src, dst)

            new_img = {
                "id": next_img_id,
                "file_name": f"images/{unique_name}",
                "width": img.get("width"),
                "height": img.get("height"),
            }
            merged["images"].append(new_img)
            img_idx[next_img_id] = {
                "path": str(new_img["file_name"]),
                "width": new_img["width"],
                "height": new_img["height"],
            }
            raw_imgid_to_new[raw_img_id] = next_img_id
            next_img_id += 1

        # Transfer annotations with category remapping
        for ann in ds.get("annotations", []):
            raw_cat = ann["category_id"]
            new_cat = raw_id_to_canonical.get(raw_cat, None)
            if new_cat is None:
                continue
            raw_img_id = ann["image_id"]
            if raw_img_id not in raw_imgid_to_new:
                continue  # image missing (e.g., if not copied)
            new_ann = {
                "id": next_ann_id,
                "image_id": raw_imgid_to_new[raw_img_id],
                "category_id": new_cat,
                "bbox": ann.get("bbox"),
                "area": ann.get("area"),
                "iscrowd": ann.get("iscrowd", 0),
            }
            # Keep segmentation if present
            if "segmentation" in ann:
                new_ann["segmentation"] = ann["segmentation"]
            merged["annotations"].append(new_ann)
            next_ann_id += 1

    # Write merged COCO for this split
    out_ann = out_split_dir / "_annotations.coco.json"
    ensure_dir(out_split_dir)
    with open(out_ann, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    return merged, img_idx

def build_multitask_csv(merged_coco: Dict, img_idx: Dict[int, Dict], out_csv: Path,
                        dirty_thresholds: Tuple[float, float] = (0.05, 0.15)):
    """
    Create CSV with columns:
      image_path, damage(0/1), dirt_present(0/1), dirt_ratio(float), clean(0/1), clean_level{0,1,2},
      damage_kinds (semicolon-separated among dent/scratch/rust)

    dirty_thresholds: (slight, strong) boundaries for ratio of dirt area to car area (or image area fallback).
    """
    # Build per-image stats
    per_img = {img_id: {"dirt_area": 0.0,
                        "car_area": 0.0,
                        "damage_kinds": set()} for img_id in img_idx.keys()}

    for ann in merged_coco.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        bbox_area = float(ann.get("area") or 0.0)
        if cat_id == CAT_ID["dirt"]:
            per_img[img_id]["dirt_area"] += bbox_area
        elif cat_id == CAT_ID["car"]:
            # Use max car area if multiple boxes
            per_img[img_id]["car_area"] = max(per_img[img_id]["car_area"], bbox_area)
        elif cat_id in (CAT_ID["dent"], CAT_ID["scratch"], CAT_ID["rust"]):
            if cat_id == CAT_ID["dent"]:
                per_img[img_id]["damage_kinds"].add("dent")
            elif cat_id == CAT_ID["scratch"]:
                per_img[img_id]["damage_kinds"].add("scratch")
            elif cat_id == CAT_ID["rust"]:
                per_img[img_id]["damage_kinds"].add("rust")

    slight, strong = dirty_thresholds
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "damage", "dirt_present", "dirt_ratio", "clean", "clean_level", "damage_kinds"])
        for img_id, meta in img_idx.items():
            path = meta["path"]
            W = meta.get("width") or 0
            H = meta.get("height") or 0
            img_area = float(W * H) if W and H else 0.0

            dirt_area = per_img[img_id]["dirt_area"]
            car_area = per_img[img_id]["car_area"] or img_area or 1.0  # fallback to img area or 1.0

            ratio = float(dirt_area) / float(car_area) if car_area > 0 else 0.0
            dirt_present = 1 if dirt_area > 0 else 0

            # Clean labels
            # clean_level: 0=clean, 1=slightly dirty, 2=strongly dirty
            if ratio >= strong:
                clean_level = 2
            elif ratio >= slight:
                clean_level = 1
            else:
                clean_level = 0
            clean = 0 if clean_level > 0 else 1  # 1=clean, 0=dirty

            damage_kinds = sorted(list(per_img[img_id]["damage_kinds"]))
            damage = 1 if damage_kinds else 0

            writer.writerow([path, damage, dirt_present, f"{ratio:.6f}", clean, clean_level, ";".join(damage_kinds)])

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to the folder containing datasets (each with train/valid/test).")
    ap.add_argument("--out", type=str, required=True, help="Output path for merged dataset.")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Which splits to merge.")
    ap.add_argument("--dirty-thresholds", nargs=2, type=float, default=[0.05, 0.15],
                    help="Thresholds for (slightly, strongly) dirty ratio relative to car area.")
    return ap.parse_args()

def main():
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out)
    ensure_dir(out_root)

    # All dataset directories inside root (skip hidden)
    dataset_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not dataset_dirs:
        print(f"[ERROR] No dataset directories found in {root}")
        return

    print("[INFO] Datasets:")
    for d in dataset_dirs:
        print("  -", d.name)

    for split in args.splits:
        print(f"[INFO] Merging split: {split}")
        merged_coco, img_idx = merge_split(split, dataset_dirs, out_root)
        csv_out = out_root / f"multitask_labels_{split}.csv"
        build_multitask_csv(merged_coco, img_idx, csv_out, tuple(args.dirty_thresholds))
        print(f"[OK] Wrote: {out_root / split / '_annotations.coco.json'}")
        print(f"[OK] Wrote: {csv_out}")

    print("\n[DONE] Merged dataset is ready.")
    print("You can train:")
    print("  - Detection (COCO) with MMDetection/Detectron2/etc.")
    print("  - Classification using the generated CSVs for cleanliness/damage heads.")

if __name__ == "__main__":
    main()
