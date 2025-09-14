#!/usr/bin/env python3
"""
prepare_dataset.py

Подготавливает датасет из папок clean/ и dirty/ в формат train/valid/test
с COCO аннотациями и CSV файлами для обучения классификации чистоты.

Структура входных данных:
    dataset/
      images/
        clean/  *.jpg
        dirty/  *.jpg
      annotations/
        _annotations.coco.json

Результирующая структура:
    dataset_prepared/
      train/
        images/  *.jpg
        _annotations.coco.json
      valid/
        images/  *.jpg  
        _annotations.coco.json
      test/
        images/  *.jpg
        _annotations.coco.json
      multitask_labels_train.csv
      multitask_labels_valid.csv
      multitask_labels_test.csv

Usage:
    python prepare_dataset.py --input ./dataset --output ./dataset_prepared --train-ratio 0.7 --valid-ratio 0.15 --test-ratio 0.15

Notes:
- Автоматически создает COCO аннотации для каждого изображения (весь кадр как bbox)
- Генерирует CSV файлы с метками чистоты для обучения
- Соблюдает баланс классов в каждом split'е
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import csv
from PIL import Image
import time


def ensure_dir(p: Path):
    """Создает директорию если её нет"""
    p.mkdir(parents=True, exist_ok=True)


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Получает размеры изображения"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"[WARN] Не удалось открыть {image_path}: {e}")
        return (640, 480)  # fallback размеры


def split_files(clean_files: List[Path], dirty_files: List[Path], 
                train_ratio: float, valid_ratio: float, test_ratio: float,
                seed: int = 42) -> Dict[str, Dict[str, List[Path]]]:
    """
    Разделяет файлы на train/valid/test с сохранением баланса классов
    """
    random.seed(seed)
    
    # Перемешиваем файлы
    clean_shuffled = clean_files.copy()
    dirty_shuffled = dirty_files.copy()
    random.shuffle(clean_shuffled)
    random.shuffle(dirty_shuffled)
    
    # Вычисляем количества для каждого split'а
    n_clean = len(clean_shuffled)
    n_dirty = len(dirty_shuffled)
    
    clean_train_count = int(n_clean * train_ratio)
    clean_valid_count = int(n_clean * valid_ratio)
    clean_test_count = n_clean - clean_train_count - clean_valid_count
    
    dirty_train_count = int(n_dirty * train_ratio)
    dirty_valid_count = int(n_dirty * valid_ratio) 
    dirty_test_count = n_dirty - dirty_train_count - dirty_valid_count
    
    # Разделяем файлы
    splits = {
        "train": {
            "clean": clean_shuffled[:clean_train_count],
            "dirty": dirty_shuffled[:dirty_train_count]
        },
        "valid": {
            "clean": clean_shuffled[clean_train_count:clean_train_count + clean_valid_count],
            "dirty": dirty_shuffled[dirty_train_count:dirty_train_count + dirty_valid_count]
        },
        "test": {
            "clean": clean_shuffled[clean_train_count + clean_valid_count:],
            "dirty": dirty_shuffled[dirty_train_count + dirty_valid_count:]
        }
    }
    
    # Выводим статистику
    for split_name, split_data in splits.items():
        clean_count = len(split_data["clean"])
        dirty_count = len(split_data["dirty"])
        total = clean_count + dirty_count
        print(f"[INFO] {split_name}: {total} изображений (clean: {clean_count}, dirty: {dirty_count})")
    
    return splits


def create_coco_annotations(images_info: List[Dict], split_name: str) -> Dict:
    """
    Создает COCO аннотации для списка изображений
    """
    categories = [
        {"id": 1, "name": "clean_car", "supercategory": "car_condition"},
        {"id": 2, "name": "dirty_car", "supercategory": "car_condition"}
    ]
    
    images = []
    annotations = []
    
    for i, img_info in enumerate(images_info):
        image_id = i + 1
        annotation_id = i + 1
        
        # Информация об изображении
        images.append({
            "id": image_id,
            "file_name": f"images/{img_info['filename']}",
            "width": img_info["width"],
            "height": img_info["height"],
            "license": 0,
            "date_captured": ""
        })
        
        # Аннотация покрывающая всё изображение
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": img_info["category_id"],
            "bbox": [0, 0, img_info["width"], img_info["height"]],
            "area": float(img_info["width"] * img_info["height"]),
            "iscrowd": 0,
            "segmentation": []
        })
    
    coco_data = {
        "info": {
            "description": f"Car Cleanliness Dataset - {split_name} split",
            "version": "1.0",
            "year": time.strftime("%Y"),
            "contributor": "Car Condition Analyzer",
            "date_created": time.strftime("%Y-%m-%d")
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations
    }
    
    return coco_data


def create_csv_labels(images_info: List[Dict], csv_path: Path):
    """
    Создает CSV файл с метками для обучения классификации
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "clean", "dirty"])
        
        for img_info in images_info:
            image_path = f"images/{img_info['filename']}"
            is_clean = 1 if img_info["category_id"] == 1 else 0
            is_dirty = 1 - is_clean
            
            writer.writerow([image_path, is_clean, is_dirty])


def copy_files_and_create_annotations(splits: Dict, input_dir: Path, output_dir: Path):
    """
    Копирует файлы и создает аннотации для каждого split'а
    """
    for split_name, split_data in splits.items():
        print(f"[INFO] Обрабатываем {split_name}...")
        
        # Создаем директории
        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        ensure_dir(images_dir)
        
        images_info = []
        
        # Обрабатываем clean изображения
        for clean_file in split_data["clean"]:
            # Копируем файл
            dest_filename = f"clean_{clean_file.name}"
            dest_path = images_dir / dest_filename
            shutil.copy2(clean_file, dest_path)
            
            # Получаем размеры
            width, height = get_image_size(clean_file)
            
            images_info.append({
                "filename": dest_filename,
                "width": width,
                "height": height,
                "category_id": 1,  # clean_car
                "original_path": str(clean_file)
            })
        
        # Обрабатываем dirty изображения
        for dirty_file in split_data["dirty"]:
            # Копируем файл
            dest_filename = f"dirty_{dirty_file.name}"
            dest_path = images_dir / dest_filename
            shutil.copy2(dirty_file, dest_path)
            
            # Получаем размеры
            width, height = get_image_size(dirty_file)
            
            images_info.append({
                "filename": dest_filename,
                "width": width,
                "height": height,
                "category_id": 2,  # dirty_car
                "original_path": str(dirty_file)
            })
        
        # Перемешиваем изображения в split'е
        random.shuffle(images_info)
        
        # Создаем COCO аннотации
        coco_data = create_coco_annotations(images_info, split_name)
        coco_path = split_dir / "_annotations.coco.json"
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        
        # Создаем CSV файл с метками
        csv_path = output_dir / f"multitask_labels_{split_name}.csv"
        create_csv_labels(images_info, csv_path)
        
        print(f"[OK] {split_name}: {len(images_info)} изображений")
        print(f"[OK] Сохранено: {coco_path}")
        print(f"[OK] Сохранено: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Подготовка датасета из папок clean/dirty в формат train/valid/test")
    parser.add_argument("--input", type=Path, required=True, help="Путь к входному датасету")
    parser.add_argument("--output", type=Path, required=True, help="Путь для сохранения подготовленного датасета")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Доля данных для обучения")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Доля данных для валидации")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Доля данных для тестирования")
    parser.add_argument("--seed", type=int, default=42, help="Random seed для воспроизводимости")
    
    args = parser.parse_args()
    
    # Проверяем, что соотношения в сумме дают 1.0
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"[ERROR] Сумма соотношений должна быть 1.0, получено: {total_ratio}")
        return
    
    input_dir = args.input
    output_dir = args.output
    
    # Проверяем структуру входных данных
    clean_dir = input_dir / "images" / "clean"
    dirty_dir = input_dir / "images" / "dirty"
    
    if not clean_dir.exists():
        print(f"[ERROR] Не найдена папка с чистыми изображениями: {clean_dir}")
        return
    
    if not dirty_dir.exists():
        print(f"[ERROR] Не найдена папка с грязными изображениями: {dirty_dir}")
        return
    
    # Собираем файлы изображений
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    clean_files = [
        f for f in clean_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    dirty_files = [
        f for f in dirty_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    print(f"[INFO] Найдено чистых изображений: {len(clean_files)}")
    print(f"[INFO] Найдено грязных изображений: {len(dirty_files)}")
    
    if len(clean_files) == 0 or len(dirty_files) == 0:
        print("[ERROR] Недостаточно изображений для разделения")
        return
    
    # Создаем выходную директорию
    ensure_dir(output_dir)
    
    # Разделяем файлы
    splits = split_files(
        clean_files, dirty_files,
        args.train_ratio, args.valid_ratio, args.test_ratio,
        args.seed
    )
    
    # Копируем файлы и создаем аннотации
    copy_files_and_create_annotations(splits, input_dir, output_dir)
    
    print(f"\n[DONE] Датасет подготовлен в: {output_dir.resolve()}")
    print("\nМожно запускать обучение:")
    print(f"python train_cleanliness.py --data-root {output_dir} --backbone convnext_tiny --epochs 20")


if __name__ == "__main__":
    main()
