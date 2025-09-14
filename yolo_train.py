#!/usr/bin/env python3
"""
Training script for YOLO damage detection model
Trains YOLOv8 model to detect car damages like dents, scratches, broken parts, etc.
"""

import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Классы повреждений
DAMAGE_CLASSES = {
    0: "damaged_door",
    1: "damaged_window", 
    2: "damaged_headlight",
    3: "damaged_mirror",
    4: "dent",
    5: "damaged_hood",
    6: "damaged_bumper",
    7: "damaged_windshield"
}

def create_dataset_config(data_dir: str, output_path: str = "config.yaml"):
    """Создает конфигурационный файл для датасета YOLO"""
    
    config = {
        'path': str(Path(data_dir).absolute()),  # Корневая папка датасета
        'train': 'images/train',  # Путь к тренировочным изображениям
        'val': 'images/val',      # Путь к валидационным изображениям
        'test': 'images/test',    # Путь к тестовым изображениям (опционально)
        
        # Количество классов
        'nc': len(DAMAGE_CLASSES),
        
        # Названия классов
        'names': list(DAMAGE_CLASSES.values())
    }
    
    # Сохраняем конфигурацию
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Конфигурация датасета сохранена в {output_path}")
    return output_path

def check_dataset_structure(data_dir: str):
    """Проверяет структуру датасета"""
    data_path = Path(data_dir)
    
    required_dirs = [
        'images/train',
        'images/val', 
        'labels/train',
        'labels/val'
    ]
    
    print("Проверка структуры датасета:")
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        exists = dir_path.exists()
        
        if exists:
            file_count = len(list(dir_path.glob('*')))
            print(f"✅ {dir_name}: {file_count} файлов")
        else:
            print(f"❌ {dir_name}: директория не найдена")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️ Неполная структура датасета. Ожидаемая структура:")
        print("""
data/
├── images/
│   ├── train/     # Тренировочные изображения (.jpg, .png)
│   └── val/       # Валидационные изображения 
└── labels/
    ├── train/     # Тренировочные аннотации (.txt)
    └── val/       # Валидационные аннотации
        """)
        return False
    
    return True

def train_yolo_model(
    data_config: str,
    model_size: str = 'yolov8n.pt',
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    lr0: float = 0.01,
    weight_decay: float = 0.0005,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
    device: str = 'auto',
    workers: int = 8,
    project: str = 'runs/detect',
    name: str = 'damage_detection',
    resume: bool = False,
    save_period: int = 10
):
    """
    Тренирует YOLO модель для детекции повреждений
    
    Args:
        data_config: Путь к конфигурации датасета
        model_size: Размер модели (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Количество эпох
        imgsz: Размер изображения для тренировки
        batch_size: Размер батча
        lr0: Начальная скорость обучения
        weight_decay: Весовая регуляризация
        mosaic: Вероятность мозаичной аугментации
        mixup: Вероятность mixup аугментации
        copy_paste: Вероятность copy-paste аугментации
        device: Устройство для тренировки ('auto', 'cpu', '0', '0,1', etc.)
        workers: Количество worker процессов для загрузки данных
        project: Папка проекта для сохранения результатов
        name: Имя эксперимента
        resume: Продолжить тренировку с последнего чекпоинта
        save_period: Период сохранения промежуточных моделей
    """
    
    print(f"🚀 Начинаем тренировку YOLO модели")
    print(f"Модель: {model_size}")
    print(f"Датасет: {data_config}")
    print(f"Эпохи: {epochs}")
    print(f"Размер изображения: {imgsz}")
    print(f"Размер батча: {batch_size}")
    print(f"Устройство: {device}")
    
    # Загружаем предтренированную модель
    model = YOLO(model_size)
    
    # Тренируем модель
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=lr0,
        weight_decay=weight_decay,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        device=device,
        workers=workers,
        project=project,
        name=name,
        resume=resume,
        save_period=save_period,
        
        # Дополнительные параметры для лучшего качества
        patience=20,          # Ранняя остановка
        optimizer='AdamW',    # Оптимизатор
        close_mosaic=10,      # Отключить мозаику за 10 эпох до конца
        amp=True,             # Automatic Mixed Precision
        plots=True,           # Сохранять графики
        val=True,             # Валидация во время тренировки
        
        # Аугментации
        degrees=10.0,         # Поворот изображения
        translate=0.1,        # Сдвиг
        scale=0.5,            # Масштабирование
        shear=0.0,            # Сдвиг
        perspective=0.0,      # Перспектива
        flipud=0.0,           # Вертикальный флип
        fliplr=0.5,           # Горизонтальный флип
        hsv_h=0.015,          # Сдвиг оттенка
        hsv_s=0.7,            # Насыщенность
        hsv_v=0.4,            # Яркость
    )
    
    print("✅ Тренировка завершена!")
    return results

def evaluate_model(model_path: str, data_config: str, imgsz: int = 640):
    """Оценивает качество натренированной модели"""
    
    print(f"📊 Оценка модели: {model_path}")
    
    # Загружаем модель
    model = YOLO(model_path)
    
    # Валидация
    metrics = model.val(data=data_config, imgsz=imgsz, plots=True)
    
    # Выводим метрики
    print(f"\n📈 Метрики модели:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # Метрики по классам
    print(f"\n📋 Метрики по классам:")
    for i, class_name in DAMAGE_CLASSES.items():
        if i < len(metrics.box.ap_class_index):
            ap = metrics.box.ap[i, 0]  # AP@0.5
            print(f"{class_name}: AP@0.5 = {ap:.4f}")
    
    return metrics

def create_training_script():
    """Создает пример скрипта для тренировки"""
    
    script_content = '''#!/usr/bin/env python3
"""
Пример скрипта тренировки YOLO модели для детекции повреждений автомобилей
"""

from yolo_train import create_dataset_config, check_dataset_structure, train_yolo_model, evaluate_model

def main():
    # Путь к датасету
    data_dir = "data"  # Замените на путь к вашему датасету
    
    # 1. Проверяем структуру датасета
    if not check_dataset_structure(data_dir):
        print("❌ Исправьте структуру датасета перед тренировкой")
        return
    
    # 2. Создаем конфигурацию датасета
    config_path = create_dataset_config(data_dir, "damage_dataset.yaml")
    
    # 3. Тренируем модель
    results = train_yolo_model(
        data_config=config_path,
        model_size='yolov8n.pt',  # Можно изменить на 's', 'm', 'l', 'x'
        epochs=100,
        imgsz=640,
        batch_size=16,
        lr0=0.01,
        device='auto',  # 'cpu' или '0' для конкретного GPU
        project='runs/damage_detection',
        name='experiment_1'
    )
    
    # 4. Оцениваем результат
    best_model = results.save_dir / 'weights' / 'best.pt'
    evaluate_model(str(best_model), config_path)

if __name__ == "__main__":
    main()
'''
    
    with open("train_damage_detection.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ Создан пример скрипта: train_damage_detection.py")

def convert_to_onnx(model_path: str, output_path: str = None):
    """Конвертирует модель в ONNX формат для деплоя"""
    
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    print(f"🔄 Конвертация модели в ONNX: {model_path} -> {output_path}")
    
    model = YOLO(model_path)
    model.export(format='onnx', imgsz=640)
    
    print(f"✅ Модель конвертирована: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Damage Detection Training')
    parser.add_argument('--data', type=str, required=True, help='Путь к датасету')
    parser.add_argument('--config', type=str, help='Путь к конфигурации (создается автоматически)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='Размер модели')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    parser.add_argument('--batch', type=int, default=16, help='Размер батча')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения')
    parser.add_argument('--lr', type=float, default=0.01, help='Скорость обучения')
    parser.add_argument('--device', type=str, default='auto', help='Устройство (auto, cpu, 0, 1, ...)')
    parser.add_argument('--project', type=str, default='runs/damage_detection', help='Папка проекта')
    parser.add_argument('--name', type=str, default=None, help='Имя эксперимента')
    parser.add_argument('--resume', action='store_true', help='Продолжить тренировку')
    parser.add_argument('--eval-only', type=str, help='Только оценка модели (путь к .pt файлу)')
    parser.add_argument('--export-onnx', type=str, help='Конвертировать модель в ONNX (путь к .pt файлу)')
    parser.add_argument('--create-example', action='store_true', help='Создать пример скрипта')
    
    args = parser.parse_args()
    
    if args.create_example:
        create_training_script()
        return
    
    if args.export_onnx:
        convert_to_onnx(args.export_onnx)
        return
    
    if args.eval_only:
        config_path = args.config or create_dataset_config(args.data)
        evaluate_model(args.eval_only, config_path)
        return
    
    # Проверяем структуру датасета
    if not check_dataset_structure(args.data):
        print("❌ Исправьте структуру датасета перед тренировкой")
        return
    
    # Создаем конфигурацию датасета
    config_path = args.config or create_dataset_config(args.data, "damage_dataset.yaml")
    
    # Генерируем имя эксперимента если не указано
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"damage_detection_{timestamp}"
    
    # Тренируем модель
    results = train_yolo_model(
        data_config=config_path,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        lr0=args.lr,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume
    )
    
    # Оцениваем результат
    best_model = results.save_dir / 'weights' / 'best.pt'
    evaluate_model(str(best_model), config_path)
    
    print(f"\n🎉 Тренировка завершена!")
    print(f"Лучшая модель: {best_model}")
    print(f"Результаты: {results.save_dir}")

if __name__ == "__main__":
    main()
