#!/usr/bin/env python3
"""
Утилиты для работы с аннотациями YOLO
Создание, конвертация и валидация аннотаций для детекции повреждений
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Классы повреждений
DAMAGE_CLASSES = {
    'damaged_door': 0,
    'damaged_window': 1, 
    'damaged_headlight': 2,
    'damaged_mirror': 3,
    'dent': 4,
    'damaged_hood': 5,
    'damaged_bumper': 6,
    'damaged_windshield': 7
}

def convert_coco_to_yolo(coco_json_path: str, images_dir: str, output_dir: str):
    """
    Конвертирует COCO аннотации в формат YOLO
    
    Args:
        coco_json_path: Путь к COCO JSON файлу
        images_dir: Папка с изображениями
        output_dir: Папка для сохранения YOLO аннотаций
    """
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Создаем маппинг ID изображений к именам файлов
    images = {img['id']: img for img in coco_data['images']}
    
    # Создаем маппинг категорий
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Группируем аннотации по изображениям
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Конвертируем каждое изображение
    for image_id, annotations in annotations_by_image.items():
        image_info = images[image_id]
        image_width = image_info['width']
        image_height = image_info['height']
        
        # Имя файла аннотации
        annotation_filename = Path(image_info['file_name']).stem + '.txt'
        annotation_path = os.path.join(output_dir, annotation_filename)
        
        yolo_annotations = []
        
        for ann in annotations:
            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Конвертируем в YOLO формат (центр + размеры, нормализованные)
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            norm_width = w / image_width
            norm_height = h / image_height
            
            # Получаем класс
            category_name = categories[ann['category_id']]
            class_id = DAMAGE_CLASSES.get(category_name, -1)
            
            if class_id != -1:
                yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Сохраняем аннотации
        with open(annotation_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        print(f"Конвертировано: {annotation_filename} ({len(yolo_annotations)} объектов)")

def convert_pascal_voc_to_yolo(xml_dir: str, output_dir: str):
    """
    Конвертирует Pascal VOC XML аннотации в формат YOLO
    
    Args:
        xml_dir: Папка с XML файлами
        output_dir: Папка для сохранения YOLO аннотаций
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    for xml_file in Path(xml_dir).glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Получаем размеры изображения
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        yolo_annotations = []
        
        # Обрабатываем каждый объект
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = DAMAGE_CLASSES.get(class_name, -1)
            
            if class_id == -1:
                print(f"Неизвестный класс: {class_name}")
                continue
            
            # Получаем bounding box
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Конвертируем в YOLO формат
            center_x = (xmin + xmax) / 2 / width
            center_y = (ymin + ymax) / 2 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        # Сохраняем аннотации
        output_file = output_dir / f"{xml_file.stem}.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        print(f"Конвертировано: {xml_file.name} -> {output_file.name} ({len(yolo_annotations)} объектов)")

def validate_yolo_annotations(images_dir: str, labels_dir: str):
    """
    Валидирует YOLO аннотации
    
    Args:
        images_dir: Папка с изображениями
        labels_dir: Папка с аннотациями
    """
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    issues = []
    stats = {'total_images': 0, 'annotated_images': 0, 'total_objects': 0}
    
    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue
        
        stats['total_images'] += 1
        
        # Проверяем наличие аннотации
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        if not label_path.exists():
            issues.append(f"Нет аннотации для изображения: {image_path.name}")
            continue
        
        stats['annotated_images'] += 1
        
        # Получаем размеры изображения
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            issues.append(f"Ошибка при открытии изображения {image_path.name}: {e}")
            continue
        
        # Проверяем аннотации
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"{label_path.name}:{line_num} - неверный формат (ожидается 5 значений, получено {len(parts)})")
                    continue
                
                try:
                    class_id, center_x, center_y, width, height = map(float, parts)
                    
                    # Проверяем класс
                    if not (0 <= class_id < len(DAMAGE_CLASSES)):
                        issues.append(f"{label_path.name}:{line_num} - неверный ID класса: {class_id}")
                    
                    # Проверяем координаты (должны быть от 0 до 1)
                    if not (0 <= center_x <= 1):
                        issues.append(f"{label_path.name}:{line_num} - center_x вне диапазона [0,1]: {center_x}")
                    if not (0 <= center_y <= 1):
                        issues.append(f"{label_path.name}:{line_num} - center_y вне диапазона [0,1]: {center_y}")
                    if not (0 < width <= 1):
                        issues.append(f"{label_path.name}:{line_num} - width вне диапазона (0,1]: {width}")
                    if not (0 < height <= 1):
                        issues.append(f"{label_path.name}:{line_num} - height вне диапазона (0,1]: {height}")
                    
                    # Проверяем что bounding box не выходит за границы
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2
                    
                    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                        issues.append(f"{label_path.name}:{line_num} - bounding box выходит за границы изображения")
                    
                    stats['total_objects'] += 1
                    
                except ValueError:
                    issues.append(f"{label_path.name}:{line_num} - не удается парсить числа: {line}")
                    
        except Exception as e:
            issues.append(f"Ошибка при чтении аннотации {label_path.name}: {e}")
    
    # Выводим результаты
    print(f"\n📊 Статистика валидации:")
    print(f"Всего изображений: {stats['total_images']}")
    print(f"Аннотированных изображений: {stats['annotated_images']}")
    print(f"Всего объектов: {stats['total_objects']}")
    print(f"Покрытие аннотациями: {stats['annotated_images']/stats['total_images']*100:.1f}%")
    
    if issues:
        print(f"\n❌ Найдено {len(issues)} проблем:")
        for issue in issues[:20]:  # Показываем первые 20
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... и еще {len(issues) - 20} проблем")
    else:
        print("\n✅ Все аннотации корректны!")
    
    return len(issues) == 0

def visualize_annotations(images_dir: str, labels_dir: str, output_dir: str, max_images: int = 10):
    """
    Визуализирует аннотации на изображениях
    
    Args:
        images_dir: Папка с изображениями
        labels_dir: Папка с аннотациями
        output_dir: Папка для сохранения визуализаций
        max_images: Максимальное количество изображений для обработки
    """
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Цвета для разных классов
    colors = [
        (255, 0, 0),    # красный
        (0, 255, 0),    # зеленый
        (0, 0, 255),    # синий
        (255, 255, 0),  # желтый
        (255, 0, 255),  # пурпурный
        (0, 255, 255),  # голубой
        (255, 128, 0),  # оранжевый
        (128, 0, 255),  # фиолетовый
    ]
    
    class_names = list(DAMAGE_CLASSES.keys())
    
    processed = 0
    
    for image_path in images_dir.iterdir():
        if processed >= max_images:
            break
        
        if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        
        # Загружаем изображение
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        height, width = image.shape[:2]
        
        # Читаем аннотации
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                class_id, center_x, center_y, bbox_width, bbox_height = map(float, line.split())
                class_id = int(class_id)
                
                # Конвертируем в пиксельные координаты
                x1 = int((center_x - bbox_width / 2) * width)
                y1 = int((center_y - bbox_height / 2) * height)
                x2 = int((center_x + bbox_width / 2) * width)
                y2 = int((center_y + bbox_height / 2) * height)
                
                # Рисуем bounding box
                color = colors[class_id % len(colors)]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Добавляем подпись
                label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            except Exception as e:
                print(f"Ошибка при обработке аннотации в {label_path.name}: {e}")
        
        # Сохраняем результат
        output_path = output_dir / f"annotated_{image_path.name}"
        cv2.imwrite(str(output_path), image)
        
        processed += 1
        print(f"Обработано: {image_path.name} -> {output_path.name}")
    
    print(f"\n✅ Создано {processed} визуализаций в папке {output_dir}")

def split_dataset(images_dir: str, labels_dir: str, output_dir: str, 
                 train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
    """
    Разделяет датасет на train/val/test наборы
    
    Args:
        images_dir: Папка с изображениями
        labels_dir: Папка с аннотациями
        output_dir: Папка для сохранения разделенного датасета
        train_ratio: Доля тренировочного набора
        val_ratio: Доля валидационного набора
        test_ratio: Доля тестового набора
    """
    
    import random
    import shutil
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Сумма долей должна быть равна 1.0"
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    # Создаем структуру папок
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Получаем список всех изображений с аннотациями
    image_files = []
    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                image_files.append(image_path.stem)
    
    # Перемешиваем
    random.shuffle(image_files)
    
    # Разделяем
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    # Копируем файлы
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for file_stem in files:
            # Находим изображение
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                if (images_dir / f"{file_stem}{ext}").exists():
                    image_file = images_dir / f"{file_stem}{ext}"
                    break
            
            if image_file is None:
                continue
            
            # Копируем изображение и аннотацию
            shutil.copy2(image_file, output_dir / 'images' / split / image_file.name)
            shutil.copy2(labels_dir / f"{file_stem}.txt", output_dir / 'labels' / split / f"{file_stem}.txt")
    
    print(f"✅ Датасет разделен:")
    print(f"  Train: {len(train_files)} изображений ({len(train_files)/total*100:.1f}%)")
    print(f"  Val: {len(val_files)} изображений ({len(val_files)/total*100:.1f}%)")
    print(f"  Test: {len(test_files)} изображений ({len(test_files)/total*100:.1f}%)")
    print(f"  Сохранено в: {output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Annotation utilities')
    parser.add_argument('command', choices=['coco2yolo', 'voc2yolo', 'validate', 'visualize', 'split'])
    parser.add_argument('--images', type=str, help='Папка с изображениями')
    parser.add_argument('--labels', type=str, help='Папка с аннотациями')
    parser.add_argument('--input', type=str, help='Входной файл/папка')
    parser.add_argument('--output', type=str, help='Выходная папка')
    parser.add_argument('--max-images', type=int, default=10, help='Максимум изображений для визуализации')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Доля тренировочного набора')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Доля валидационного набора')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Доля тестового набора')
    
    args = parser.parse_args()
    
    if args.command == 'coco2yolo':
        convert_coco_to_yolo(args.input, args.images, args.output)
    elif args.command == 'voc2yolo':
        convert_pascal_voc_to_yolo(args.input, args.output)
    elif args.command == 'validate':
        validate_yolo_annotations(args.images, args.labels)
    elif args.command == 'visualize':
        visualize_annotations(args.images, args.labels, args.output, args.max_images)
    elif args.command == 'split':
        split_dataset(args.images, args.labels, args.output, 
                     args.train_ratio, args.val_ratio, args.test_ratio)

if __name__ == "__main__":
    main()
