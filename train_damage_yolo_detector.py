#!/usr/bin/env python3
"""
Простой пример тренировки YOLO модели для детекции повреждений автомобилей
Использование: python train_damage_simple.py
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

def create_dataset_config():
    """Создает конфигурацию датасета"""
    config = {
        'path': str(Path('data').absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 8,  # количество классов
        'names': [
            'damaged_door',
            'damaged_window', 
            'damaged_headlight',
            'damaged_mirror',
            'dent',
            'damaged_hood',
            'damaged_bumper',
            'damaged_windshield'
        ]
    }
    
    with open('damage_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return 'damage_config.yaml'

def main():
    # Создаем конфигурацию
    config_path = create_dataset_config()
    
    # Загружаем предтренированную модель YOLOv8
    model = YOLO('yolov8n.pt')  # nano версия для быстрой тренировки
    
    # Тренируем модель
    results = model.train(
        data=config_path,
        epochs=50,          # Количество эпох
        imgsz=640,          # Размер изображения
        batch=8,            # Размер батча (уменьшите если мало памяти)
        lr0=0.01,          # Скорость обучения
        device='auto',      # Автоматический выбор устройства
        project='runs',     # Папка для результатов
        name='damage_detection',
        
        # Настройки для лучшего качества
        patience=10,        # Ранняя остановка
        save_period=10,     # Сохранять каждые 10 эпох
        plots=True,         # Сохранять графики
        val=True,          # Валидация
        
        # Аугментации данных
        flipud=0.0,        # Вертикальный поворот
        fliplr=0.5,        # Горизонтальный поворот
        degrees=10,        # Поворот изображения
        translate=0.1,     # Сдвиг
        scale=0.5,         # Масштабирование
        hsv_h=0.015,       # Сдвиг цвета
        hsv_s=0.7,         # Насыщенность
        hsv_v=0.4,         # Яркость
    )
    
    print("✅ Тренировка завершена!")
    print(f"Результаты сохранены в: {results.save_dir}")
    print(f"Лучшая модель: {results.save_dir}/weights/best.pt")
    
    # Валидация модели
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val()
    
    print("\n📊 Метрики качества:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()
