#!/usr/bin/env python3
"""
Scrape ONLY openly-licensed images (CC0/CC BY/CC BY-SA/Public Domain) for two classes:
  - "dirty" cars
  - "clean" cars

Sources (all return license metadata):
  - Openverse API (no key; aggregates CC sources)
  - Wikimedia Commons API (all free content; we still filter to PD/CC0/CC BY/CC BY-SA)
  - Flickr API (needs FLICKR_API_KEY; filter to commercial-friendly CC licenses only)

Each saved image gets COCO annotation with a single full-image bbox categorized as
`dirty_car` or `clean_car`. COCO 'licenses' are filled with the actual license records,
and each image references the matching license id. Extra fields `source_url` and `author`
are added to `images` entries for attribution.
"""
import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import requests
from PIL import Image, UnidentifiedImageError
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
    "waxed car",
    "detailing finish",
]


ALLOWED_LICENSE_KEYS = {"CC0", "CC-BY", "CC-BY-SA", "PD"}

def canonical_license(name_or_code: str) -> Optional[str]:
    s = (name_or_code or "").strip().lower()
    # Openverse
    if s in {"cc0"}:
        return "CC0"
    if s in {"by"}:
        return "CC-BY"
    if s in {"by-sa"}:
        return "CC-BY-SA"
    if s in {"pdm"}:
        return "PD"
    # Wikimedia Commons
    if "cc0" in s:
        return "CC0"
    if re.search(r"\bcc[-\s]?by\b", s):
        return "CC-BY"
    if re.search(r"\bcc[-\s]?by[-\s]?sa\b", s):
        return "CC-BY-SA"
    if "public domain" in s or "pd-" in s or s == "pd":
        return "PD"
    if "no known copyright restrictions" in s or "us gov" in s or "u.s. gov" in s:
        return "PD"
    # Flickr codes
    if s in {"4"}:   # CC BY
        return "CC-BY"
    if s in {"5"}:   # CC BY-SA
        return "CC-BY-SA"
    if s in {"7","8","10"}:  # PD-ish
        return "PD"
    if s in {"9"}:   # CC0
        return "CC0"
    return None


@dataclass
class Candidate:
    url: str
    source_url: str
    author: str
    license_key: str


@dataclass
class ImageMeta:
    file_path: str
    width: int
    height: int
    category_id: int
    image_id: int
    annotation_id: int
    license_id: int
    author: str
    source_url: str


async def fetch_bytes(session: aiohttp.ClientSession, url: str, timeout_s: int = 25) -> Optional[bytes]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                return None
            ctype = resp.headers.get("Content-Type", "").lower()
            if "image" not in ctype and not re.search(r"\.(jpg|jpeg|png|webp|bmp|tiff?)($|\?)", url, re.I):
                return None
            return await resp.read()
    except Exception:
        return None


def perceptual_dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    diff_bits = []
    for y in range(size):
        for x in range(size):
            diff_bits.append(g.getpixel((x, y)) > g.getpixel((x + 1, y)))
    v = 0
    for i, bit in enumerate(diff_bits):
        if bit:
            v |= 1 << i
    return v


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)[:120]


def gather_openverse(queries: List[str], target: int) -> List[Candidate]:
    base = "https://api.openverse.org/v1/images/"
    page_size = 200
    candidates: List[Candidate] = []
    seen_urls: Set[str] = set()
    for q in queries:
        page = 1
        while len(candidates) < target and page <= 50:
            params = {
                "q": q,
                "image_type": "photo",
                "page_size": page_size,
                "page": page,
                "license_type": "commercial,modification",
            }
            try:
                r = requests.get(base, params=params, timeout=30)
                if r.status_code != 200:
                    break
                data = r.json()
                results = data.get("results", [])
                if not results:
                    break
                for item in results:
                    lic_raw = (item.get("license") or "").lower().strip()
                    key = canonical_license(lic_raw)
                    if key is None:
                        continue
                    url = item.get("url") or item.get("thumbnail") or ""
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    author = item.get("creator") or item.get("creator_url") or ""
                    src = item.get("foreign_landing_url") or item.get("url") or ""
                    candidates.append(Candidate(url=url, source_url=src, author=author, license_key=key))
                    if len(candidates) >= target:
                        break
                page += 1
            except Exception:
                break
            if len(results) < page_size:
                break
        if len(candidates) >= target:
            break
    return candidates


def gather_wikicommons(queries: List[str], target: int) -> List[Candidate]:
    endpoint = "https://commons.wikimedia.org/w/api.php"
    candidates: List[Candidate] = []
    for q in queries:
        cont = {}
        while len(candidates) < target:
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": q,
                "gsrlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 1024,
                **cont
            }
            try:
                r = requests.get(endpoint, params=params, timeout=30)
                if r.status_code != 200:
                    break
                data = r.json()
                pages = (data.get("query") or {}).get("pages", {})
                if not pages:
                    break
                for _, page in pages.items():
                    infos = page.get("imageinfo") or []
                    if not infos:
                        continue
                    info = infos[0]
                    url = info.get("url") or ""
                    ext = info.get("extmetadata") or {}
                    lic_short = (ext.get("LicenseShortName", {}) or {}).get("value", "")
                    lic_url = (ext.get("LicenseUrl", {}) or {}).get("value", "")
                    artist = (ext.get("Artist", {}) or {}).get("value", "")
                    key = canonical_license(lic_short) or canonical_license(lic_url)
                    if key is None:
                        continue
                    title = page.get("title","")
                    src = f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"
                    author = re.sub("<[^<]+?>", "", artist).strip()
                    candidates.append(Candidate(url=url, source_url=src, author=author, license_key=key))
                    if len(candidates) >= target:
                        break
                cont = data.get("continue") or {}
                if not cont:
                    break
            except Exception:
                break
        if len(candidates) >= target:
            break
    return candidates


def gather_flickr(queries: List[str], target: int, api_key: str) -> List[Candidate]:
    endpoint = "https://api.flickr.com/services/rest"
    licenses_allowed = "4,5,7,8,9,10"
    per_page = 250
    candidates: List[Candidate] = []
    seen_ids: Set[str] = set()
    for q in queries:
        page = 1
        while len(candidates) < target and page <= 40:
            params = {
                "method": "flickr.photos.search",
                "api_key": api_key,
                "text": q,
                "content_type": 1,
                "media": "photos",
                "safe_search": 1,
                "license": licenses_allowed,
                "per_page": per_page,
                "page": page,
                "extras": "url_o,url_l,url_c,url_m,owner_name,license",
                "format": "json",
                "nojsoncallback": 1,
            }
            try:
                r = requests.get(endpoint, params=params, timeout=30)
                if r.status_code != 200:
                    break
                data = r.json()
                photos = (data.get("photos") or {}).get("photo", [])
                if not photos:
                    break
                for p in photos:
                    pid = str(p.get("id"))
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    url = p.get("url_o") or p.get("url_l") or p.get("url_c") or p.get("url_m")
                    if not url:
                        continue
                    lic_code = str(p.get("license",""))
                    key = canonical_license(lic_code)
                    if key is None:
                        continue
                    owner = p.get("ownername") or ""
                    src = f"https://www.flickr.com/photo.gne?id={pid}"
                    candidates.append(Candidate(url=url, source_url=src, author=owner, license_key=key))
                    if len(candidates) >= target:
                        break
                if len(photos) < per_page:
                    break
                page += 1
            except Exception:
                break
        if len(candidates) >= target:
            break
    return candidates


async def download_and_filter(
    candidates: List[Candidate],
    out_dir: Path,
    keep: int,
    min_side: int,
    resize_longer: int,
    concurrency: int = 32,
    global_dhash: Optional[Set[int]] = None,
    global_sha1: Optional[Set[str]] = None,
) -> List[Tuple[Path, Candidate, int, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Tuple[Path, Candidate, int, int]] = []
    sem = asyncio.Semaphore(concurrency)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; car-coco-cc/1.0)"}
    global_dhash = global_dhash or set()
    global_sha1 = global_sha1 or set()

    async def worker(c: Candidate):
        nonlocal saved
        if len(saved) >= keep:
            return
        async with sem:
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    data = await fetch_bytes(session, c.url)
                if not data:
                    return
                import hashlib
                sha1 = hashlib.sha1(data).hexdigest()
                if sha1 in global_sha1:
                    return
                try:
                    img = Image.open(BytesIO(data)).convert("RGB")
                except UnidentifiedImageError:
                    return
                w, h = img.size
                if w < min_side or h < min_side:
                    return
                if resize_longer and max(w, h) > resize_longer:
                    if w >= h:
                        new_w = resize_longer
                        new_h = int(h * (resize_longer / w))
                    else:
                        new_h = resize_longer
                        new_w = int(w * (resize_longer / h))
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    w, h = img.size
                # dHash dedup
                g = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                dh_bits = [(g.getpixel((x, y)) > g.getpixel((x + 1, y))) for y in range(8) for x in range(8)]
                dh = 0
                for i, bit in enumerate(dh_bits):
                    if bit:
                        dh |= 1 << i
                if dh in global_dhash:
                    return
                fname = sanitize_filename(f"{c.license_key}_{sha1[:12]}.jpg")
                path = out_dir / fname
                img.save(path, "JPEG", quality=90, optimize=True)
                global_sha1.add(sha1)
                global_dhash.add(dh)
                saved.append((path, c, w, h))
            except Exception:
                return

    tasks = [asyncio.create_task(worker(c)) for c in candidates]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Downloading -> {out_dir.name}"):
        await f
        if len(saved) >= keep:
            break
    return saved[:keep]


def build_coco(items: List[ImageMeta], license_index: Dict[str, int], license_records: Dict[str, Dict[str, str]], root_dir: Path) -> Dict:
    images, annotations = [], []
    for m in items:
        rel = os.path.relpath(m.file_path, root_dir).replace(os.sep, "/")
        images.append({
            "id": m.image_id,
            "file_name": rel,
            "width": m.width,
            "height": m.height,
            "license": m.license_id,
            "date_captured": "",
            "author": m.author,
            "source_url": m.source_url,
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
    licenses = []
    for key, lid in sorted(license_index.items(), key=lambda kv: kv[1]):
        rec = license_records[key]
        licenses.append({"id": lid, "name": rec.get("name", key), "url": rec.get("url", "")})
    categories = [
        {"id": 1, "name": "dirty_car", "supercategory": "car_condition"},
        {"id": 2, "name": "clean_car", "supercategory": "car_condition"},
    ]
    return {"info": {"description": "Dirty vs Clean Cars (CC/PD only)", "version": "1.0",
                     "year": time.strftime("%Y"), "date_created": time.strftime("%Y-%m-%d")},
            "licenses": licenses, "categories": categories, "images": images, "annotations": annotations}


def main():
    parser = argparse.ArgumentParser(description="Scrape CC/PD-only images (dirty vs clean cars) and export COCO.")
    parser.add_argument("--out", type=Path, default=Path("dataset_cc"), help="Output dataset directory.")
    parser.add_argument("--per-class", type=int, default=500, help="Target images per class.")
    parser.add_argument("--sources", type=str, nargs="+", default=["openverse", "wikicommons"],
                        choices=["openverse", "wikicommons", "flickr"], help="Data sources to use.")
    parser.add_argument("--min-side", type=int, default=256, help="Minimum width/height allowed.")
    parser.add_argument("--resize-longer", type=int, default=0, help="Resize longest side to this size (0 = keep).")
    parser.add_argument("--concurrency", type=int, default=32, help="Concurrent downloads.")
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
    target = args.per_class

    dirty_candidates: List[Candidate] = []
    clean_candidates: List[Candidate] = []

    per_source_target = max(300, target)  # overfetch per source

    if "openverse" in args.sources:
        dirty_candidates += gather_openverse(dirty_queries, per_source_target)
        clean_candidates += gather_openverse(clean_queries, per_source_target)

    if "wikicommons" in args.sources:
        dirty_candidates += gather_wikicommons(dirty_queries, per_source_target)
        clean_candidates += gather_wikicommons(clean_queries, per_source_target)

    if "flickr" in args.sources:
        key = os.environ.get("FLICKR_API_KEY")
        if not key:
            print("WARNING: FLICKR_API_KEY not set; skipping Flickr.")
        else:
            dirty_candidates += gather_flickr(dirty_queries, per_source_target, api_key=key)
            clean_candidates += gather_flickr(clean_queries, per_source_target, api_key=key)

    license_records = {
        "CC0":    {"name": "CC0 1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"},
        "CC-BY":  {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
        "CC-BY-SA":{"name": "CC BY-SA 4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"},
        "PD":     {"name": "Public Domain", "url": "https://creativecommons.org/publicdomain/mark/1.0/"},
    }
    license_index = {k: i+1 for i, k in enumerate(license_records.keys())}

    global_dhash: Set[int] = set()
    global_sha1: Set[str] = set()

    saved_dirty = asyncio.run(download_and_filter(dirty_candidates, dirty_dir, keep=target,
                                                  min_side=args.min_side, resize_longer=args.resize_longer,
                                                  concurrency=args.concurrency,
                                                  global_dhash=global_dhash, global_sha1=global_sha1))
    saved_clean = asyncio.run(download_and_filter(clean_candidates, clean_dir, keep=target,
                                                  min_side=args.min_side, resize_longer=args.resize_longer,
                                                  concurrency=args.concurrency,
                                                  global_dhash=global_dhash, global_sha1=global_sha1))

    metas: List[ImageMeta] = []
    next_img, next_ann = 1, 1

    def push(saved_list, category_id: int):
        nonlocal next_img, next_ann
        for path, cand, w, h in saved_list:
            lic_id = license_index[cand.license_key]
            metas.append(ImageMeta(str(path), w, h, category_id, next_img, next_ann,
                                   lic_id, cand.author, cand.source_url))
            next_img += 1
            next_ann += 1

    push(saved_dirty, 1)
    push(saved_clean, 2)

    coco = build_coco(metas, license_index, license_records, root_dir=out_root)
    anno_path = anno_dir / "instances.json"
    with open(anno_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print("\nDone!")
    print(f"Dataset root: {out_root.resolve()}")
    print(f"Images root: {images_root.resolve()}")
    print(f"Annotations: {anno_path.resolve()}")
    print(f"Final counts -> dirty: {sum(1 for m in metas if m.category_id == 1)}, clean: {sum(1 for m in metas if m.category_id == 2)}")


if __name__ == "__main__":
    main()
