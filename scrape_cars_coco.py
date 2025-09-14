#!/usr/bin/env python3
"""
Scrape images for two classes ("dirty" cars and "clean" cars) and produce a COCO-style
annotation file. Images are saved into separate folders and each image gets a single
full-image bounding box labeled with its class (sufficient for classification workflows
that expect COCO format).

⚠️ Notes & Ethics
- This script uses DuckDuckGo image search results via `duckduckgo_search` and downloads
  the images directly from third‑party sites. Respect each site's Terms of Service and
  copyright/licensing. Use for personal research/training; do not redistribute images
  without proper rights.
- Image search quality varies; some results will be off-topic. The script deduplicates
  by perceptual hash and filters out tiny/corrupt images, but manual curation is still helpful.

Usage:
    python scrape_cars_coco.py --per-class 500 --out ./dataset --concurrency 32

Optional:
    python scrape_cars_coco.py --per-class 300 --out ./dataset --min-side 256 --resize-longer 0

Requirements (install first):
    pip install -r requirements.txt

Resulting structure:
    dataset/
      images/
        dirty/  *.jpg
        clean/  *.jpg
      annotations/
        instances.json   # COCO annotations for all images

COCO labeling approach:
- categories: {1: "dirty_car"}, {2: "clean_car"} under supercategory "car_condition"
- For each image, we create a single annotation whose bbox covers the whole image,
  i.e. [0, 0, width, height], with area = width*height, iscrowd = 0, segmentation = []
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import aiohttp
from PIL import Image, UnidentifiedImageError
from duckduckgo_search import DDGS
from tqdm import tqdm


DEFAULT_DIRTY_QUERIES = [
    "dirty car",
    "muddy car",
    "mud covered car",
    "mud-splattered car",
    "dusty car",
    "offroad mud car",
    "rally car mud",
    "car after offroad",
    "dirty vehicle",
    "muddy suv",
    "muddy truck",
    "car covered in dirt",
]

DEFAULT_CLEAN_QUERIES = [
    "clean car",
    "shiny car",
    "polished car",
    "car after car wash",
    "freshly washed car",
    "new car showroom",
    "detailed car paint",
    "clean vehicle",
    "car detailing result",
    "waxed car",
]


@dataclass
class ImageMeta:
    file_name: str
    width: int
    height: int
    category_id: int
    image_id: int
    annotation_id: int


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    # prevent overly long names
    return name[:120]


def perceptual_dhash(img: Image.Image, size: int = 8) -> int:
    """Compute a simple difference hash (dHash) for deduplication."""
    # convert to grayscale and shrink
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    diff_bits = []
    for y in range(size):
        for x in range(size):
            diff_bits.append(g.getpixel((x, y)) > g.getpixel((x + 1, y)))
    value = 0
    for i, bit in enumerate(diff_bits):
        if bit:
            value |= 1 << i
    return value


async def ddg_image_urls(
    queries: List[str],
    target_count: int,
    per_query_batch: int = 100,
    safesearch: str = "moderate",
) -> List[str]:
    """
    Pull image URLs from DuckDuckGo for the given queries until we collect target_count unique URLs.
    """
    urls: Set[str] = set()
    random.shuffle(queries)

    with DDGS() as ddgs:
        # We loop over queries repeatedly until we reach the target or plateau.
        plateau_rounds = 0
        previous_len = 0
        while len(urls) < target_count and plateau_rounds < 3:
            for q in queries:
                needed = target_count - len(urls)
                if needed <= 0:
                    break
                try:
                    results = ddgs.images(
                        keywords=q,
                        max_results=min(per_query_batch, needed),
                        safesearch=safesearch,  # "off" | "moderate" | "on"
                        type_image="photo",      # prefer photographs
                    )
                    for r in results:
                        u = r.get("image") or r.get("thumbnail") or r.get("url")
                        if not u:
                            continue
                        if u.startswith("data:"):
                            continue
                        urls.add(u)
                except Exception:
                    # If DDG throttles or errors out, continue with next query
                    continue

            if len(urls) == previous_len:
                plateau_rounds += 1
            else:
                plateau_rounds = 0
                previous_len = len(urls)

    return list(urls)


async def fetch_bytes(session: aiohttp.ClientSession, url: str, timeout_s: int = 20) -> Optional[bytes]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                return None
            ctype = resp.headers.get("Content-Type", "").lower()
            if "image" not in ctype:
                # Some hosts don't send content-type; allow anyway if extension looks like image
                if not re.search(r"\.(jpg|jpeg|png|webp|bmp|tiff?)($|\?)", url, re.I):
                    return None
            return await resp.read()
    except Exception:
        return None


async def download_class(
    class_name: str,
    category_id: int,
    urls: List[str],
    out_dir: Path,
    start_image_id: int,
    start_ann_id: int,
    concurrency: int = 32,
    min_side: int = 256,
    resize_longer: int = 0,  # 0 = keep original
    dedup_dhash: Set[int] = None,
    dedup_sha1: Set[str] = None,
) -> Tuple[List[ImageMeta], int, int, int]:
    """
    Download images for a single class and create ImageMeta list.
    Returns (metas, next_image_id, next_ann_id, saved_count)
    """
    dedup_dhash = dedup_dhash if dedup_dhash is not None else set()
    dedup_sha1 = dedup_sha1 if dedup_sha1 is not None else set()

    out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; car-coco-scraper/1.0; +https://duckduckgo.com)"
    }

    async def worker(url: str) -> Optional[ImageMeta]:
        nonlocal start_image_id, start_ann_id
        async with sem:
            # Session is created outside
            try:
                data = await fetch_bytes(session, url)
                if not data:
                    return None

                # Check duplicate by sha1 on raw bytes
                sha1 = hashlib.sha1(data).hexdigest()
                if sha1 in dedup_sha1:
                    return None

                # Load image
                try:
                    img = Image.open(BytesIO(data))
                    img = img.convert("RGB")
                except UnidentifiedImageError:
                    return None

                w, h = img.size
                if w < min_side or h < min_side:
                    return None

                # Optional resize (keep aspect ratio) by longest side
                if resize_longer and max(w, h) > resize_longer:
                    if w >= h:
                        new_w = resize_longer
                        new_h = int(h * (resize_longer / w))
                    else:
                        new_h = resize_longer
                        new_w = int(w * (resize_longer / h))
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    w, h = img.size

                # Perceptual hash deduplication
                dh = perceptual_dhash(img)
                if dh in dedup_dhash:
                    return None

                # Save
                fname_base = sanitize_filename(f"{class_name}_{sha1[:12]}.jpg")
                out_path = out_dir / fname_base
                img.save(out_path, format="JPEG", quality=90, optimize=True)

                # Update dedup sets only after successful save
                dedup_sha1.add(sha1)
                dedup_dhash.add(dh)

                image_id = start_image_id
                ann_id = start_ann_id
                start_image_id += 1
                start_ann_id += 1

                return ImageMeta(
                    file_name=str(out_path),
                    width=w,
                    height=h,
                    category_id=category_id,
                    image_id=image_id,
                    annotation_id=ann_id,
                )
            except Exception:
                return None

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [asyncio.create_task(worker(u)) for u in urls]
        metas: List[ImageMeta] = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Downloading {class_name}"):
            r = await f
            if r is not None:
                metas.append(r)

    return metas, start_image_id, start_ann_id, len(metas)


def build_coco(metas: List[ImageMeta], root_dir: Path) -> Dict:
    # Make file_name relative to the dataset root dir
    images = []
    annotations = []

    for m in metas:
        rel_path = os.path.relpath(m.file_name, root_dir)
        images.append({
            "id": m.image_id,
            "file_name": rel_path.replace(os.sep, "/"),
            "width": m.width,
            "height": m.height,
            "license": 0,
            "date_captured": "",
        })
        annotations.append({
            "id": m.annotation_id,
            "image_id": m.image_id,
            "category_id": m.category_id,
            "bbox": [0, 0, m.width, m.height],
            "area": float(m.width * m.height),
            "iscrowd": 0,
            "segmentation": [],
        })

    categories = [
        {"id": 1, "name": "dirty_car", "supercategory": "car_condition"},
        {"id": 2, "name": "clean_car", "supercategory": "car_condition"},
    ]

    coco = {
        "info": {
            "description": "Dirty vs Clean Cars (auto-scraped)",
            "version": "1.0",
            "year": time.strftime("%Y"),
            "contributor": "",
            "date_created": time.strftime("%Y-%m-%d"),
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    return coco


def main():
    parser = argparse.ArgumentParser(description="Scrape dirty/clean car images and export COCO annotations.")
    parser.add_argument("--out", type=Path, default=Path("dataset"), help="Output dataset directory.")
    parser.add_argument("--per-class", type=int, default=500, help="Target images per class.")
    parser.add_argument("--concurrency", type=int, default=32, help="Concurrent downloads.")
    parser.add_argument("--min-side", type=int, default=256, help="Minimum width/height allowed.")
    parser.add_argument("--resize-longer", type=int, default=0, help="Resize longest side to this size (0 = keep).")
    parser.add_argument("--safesearch", type=str, default="moderate", choices=["off", "moderate", "on"], help="DuckDuckGo safesearch.")
    parser.add_argument("--dirty-queries", type=str, nargs="*", default=None, help="Override dirty queries.")
    parser.add_argument("--clean-queries", type=str, nargs="*", default=None, help="Override clean queries.")
    args = parser.parse_args()

    out_root = args.out
    images_root = out_root / "images"
    dirty_dir = images_root / "dirty"
    clean_dir = images_root / "clean"
    anno_dir = out_root / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)

    dirty_queries = args.dirty_queries or DEFAULT_DIRTY_QUERIES
    clean_queries = args.clean_queries or DEFAULT_CLEAN_QUERIES

    target_per_class = args.per_class

    # 1) Search URLs for each class (overfetch a bit to offset failures)
    overfetch = int(target_per_class * 1.8)
    print(f"Collecting URLs (dirty): target={target_per_class}, overfetch={overfetch}")
    dirty_urls = asyncio.run(ddg_image_urls(dirty_queries, target_count=overfetch, safesearch=args.safesearch))
    print(f"Got {len(dirty_urls)} candidate URLs for 'dirty'.")

    print(f"Collecting URLs (clean): target={target_per_class}, overfetch={overfetch}")
    clean_urls = asyncio.run(ddg_image_urls(clean_queries, target_count=overfetch, safesearch=args.safesearch))
    print(f"Got {len(clean_urls)} candidate URLs for 'clean'.")

    # 2) Download with concurrency + dedup across both classes
    start_image_id = 1
    start_ann_id = 1
    dedup_dhash: Set[int] = set()
    dedup_sha1: Set[str] = set()

    metas_all: List[ImageMeta] = []

    metas_dirty, start_image_id, start_ann_id, saved_dirty = asyncio.run(
        download_class(
            class_name="dirty",
            category_id=1,
            urls=dirty_urls,
            out_dir=dirty_dir,
            start_image_id=start_image_id,
            start_ann_id=start_ann_id,
            concurrency=args.concurrency,
            min_side=args.min_side,
            resize_longer=args.resize_longer,
            dedup_dhash=dedup_dhash,
            dedup_sha1=dedup_sha1,
        )
    )
    metas_all.extend(metas_dirty)
    print(f"Saved {saved_dirty} 'dirty' images.")

    # If we saved too few, try again with more URLs (search another round) until we plateau
    retry_rounds = 0
    while saved_dirty < target_per_class and retry_rounds < 2:
        missing = target_per_class - saved_dirty
        print(f"Retrying 'dirty' to fill {missing} more...")
        more_dirty = asyncio.run(ddg_image_urls(dirty_queries, target_count=missing + 50, safesearch=args.safesearch))
        metas_dirty2, start_image_id, start_ann_id, saved_dirty2 = asyncio.run(
            download_class(
                class_name="dirty",
                category_id=1,
                urls=more_dirty,
                out_dir=dirty_dir,
                start_image_id=start_image_id,
                start_ann_id=start_ann_id,
                concurrency=args.concurrency,
                min_side=args.min_side,
                resize_longer=args.resize_longer,
                dedup_dhash=dedup_dhash,
                dedup_sha1=dedup_sha1,
            )
        )
        metas_all.extend(metas_dirty2)
        saved_dirty += saved_dirty2
        retry_rounds += 1
        print(f"Total 'dirty' now: {saved_dirty}")

    metas_clean, start_image_id, start_ann_id, saved_clean = asyncio.run(
        download_class(
            class_name="clean",
            category_id=2,
            urls=clean_urls,
            out_dir=clean_dir,
            start_image_id=start_image_id,
            start_ann_id=start_ann_id,
            concurrency=args.concurrency,
            min_side=args.min_side,
            resize_longer=args.resize_longer,
            dedup_dhash=dedup_dhash,
            dedup_sha1=dedup_sha1,
        )
    )
    metas_all.extend(metas_clean)
    print(f"Saved {saved_clean} 'clean' images.")

    retry_rounds = 0
    while saved_clean < target_per_class and retry_rounds < 2:
        missing = target_per_class - saved_clean
        print(f"Retrying 'clean' to fill {missing} more...")
        more_clean = asyncio.run(ddg_image_urls(clean_queries, target_count=missing + 50, safesearch=args.safesearch))
        metas_clean2, start_image_id, start_ann_id, saved_clean2 = asyncio.run(
            download_class(
                class_name="clean",
                category_id=2,
                urls=more_clean,
                out_dir=clean_dir,
                start_image_id=start_image_id,
                start_ann_id=start_ann_id,
                concurrency=args.concurrency,
                min_side=args.min_side,
                resize_longer=args.resize_longer,
                dedup_dhash=dedup_dhash,
                dedup_sha1=dedup_sha1,
            )
        )
        metas_all.extend(metas_clean2)
        saved_clean += saved_clean2
        retry_rounds += 1
        print(f"Total 'clean' now: {saved_clean}")

    # Optionally trim to exactly target_per_class per class (by newest first)
    def trim_to_target(metas: List[ImageMeta], class_id: int, target: int) -> List[ImageMeta]:
        subset = [m for m in metas if m.category_id == class_id]
        others = [m for m in metas if m.category_id != class_id]
        if len(subset) > target:
            subset = subset[:target]
        return subset + others

    metas_all = trim_to_target(metas_all, 1, target_per_class)
    metas_all = trim_to_target(metas_all, 2, target_per_class)

    # 3) Build and save COCO
    coco = build_coco(metas_all, root_dir=out_root)
    anno_path = anno_dir / "instances.json"
    with open(anno_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print("\nDone!")
    print(f"Dataset root: {out_root.resolve()}")
    print(f"Images: {images_root.resolve()}")
    print(f"Annotations: {anno_path.resolve()}")
    print(f"Final counts -> dirty: {sum(1 for m in metas_all if m.category_id == 1)}, clean: {sum(1 for m in metas_all if m.category_id == 2)}")


if __name__ == "__main__":
    main()
