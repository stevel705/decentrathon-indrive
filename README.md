# 🚗 Integrated Car Condition Analysis System

Комплексная система анализа состояния автомобилей, сочетающая детекцию повреждений с помощью YOLOv8 и классификацию чистоты с использованием сверточных нейронных сетей.

> **Современная система для автосервисов, страховых компаний и платформ продажи автомобилей**

## 🎯 Основные возможности

- **🔧 Детекция повреждений**: Автоматическое обнаружение различных типов повреждений (вмятины, царапины, поврежденные фары, двери и т.д.) с помощью YOLOv8
- **🧽 Анализ чистоты**: Бинарная классификация состояния автомобиля (чистый/грязный) с помощью ConvNeXt/EfficientNet
- **🌐 Веб-интерфейс**: Удобный FastAPI веб-сервис с HTML/JavaScript интерфейсом
- **📊 Комплексная аналитика**: Детальные отчеты с визуализацией результатов
- **🎓 Специализированное обучение**: Отдельные скрипты для обучения моделей детекции повреждений и классификации чистоты

## 🖥️ Системные требования

### Минимальные требования
- **Python**: 3.11+
- **RAM**: 8GB (для обучения рекомендуется 16GB+)
- **Диск**: 5GB свободного места
- **ОС**: Linux, macOS, Windows

### Рекомендуемые требования
- **GPU**: NVIDIA с поддержкой CUDA (для ускорения обучения)
- **RAM**: 16GB+
- **Диск**: SSD с 20GB+ свободного места
- **CPU**: 8+ ядер

### Зависимости
- **uv**: Современный менеджер пакетов Python
- **PyTorch**: 2.4.1+ (с CUDA поддержкой при наличии GPU)
- **timm**: Для backbone моделей
- **FastAPI**: Для веб-интерфейса
- **Ultralytics**: Для YOLOv8

## 🎯 Основные возможности

## 📁 Структура проекта

```
decentraton_indrive/
├── 🌐 Веб-приложение
│   └── app.py                    # Главное интегрированное приложение (FastAPI)
│
├── 🎓 Обучение моделей
│   ├── train_cleanliness.py      # Обучение модели чистоты (бинарная классификация)
│   ├── train_damage_detection.py # Обучение модели детекции повреждений
│   └── maskrcnn_train_eval.py    # Обучение Mask R-CNN для сегментации
│
├── 📊 Подготовка данных
│   └── scripts/                  # Скрипты подготовки данных
│       ├── prepare_dataset.py    # Подготовка датасета из папок clean/dirty
│       ├── merge_and_label_coco.py # Слияние COCO аннотаций
│       └── scrape_cars_coco.py   # Скрапинг данных
│
├── 🔍 Инференс
│   ├── predict.py                # Универсальные предсказания моделей
│   ├── predict_examples.py       # Примеры использования predict.py
│   └── yolo_damage_detector.py   # Специализированная YOLO детекция
│
├── 🛠️ Утилиты
│   └── health_check.py           # Проверка состояния системы
│
├── 📂 Данные и модели
│   ├── data/                     # Исходные данные
│   ├── dataset/                  # Сырые данные для обработки
│   ├── dataset_prepared/         # Подготовленные данные для обучения
│   ├── weights/                  # Обученные модели ⬇️ СКАЧАТЬ С GOOGLE DRIVE
│   │   ├── dirt_weights.pt       # Модель классификации чистоты
│   │   └── best_yolo_damage.pt   # YOLO модель для повреждений
│   └── runs/                     # Результаты обучения
|
└── ⚙️ Конфигурация
    ├── pyproject.toml            # Основные зависимости проекта (uv)
    ├── requirements_web.txt      # Дополнительные веб-зависимости
    ├── uv.lock                   # Файл блокировки зависимостей
    └── README.md                 # Документация проекта
```

## 🚀 Быстрый старт

### 0. Загрузка предобученных моделей

**⬇️ Скачайте обученные веса моделей:**

**Основной архив с весами:**
- 📥 **Google Drive**: [car_analysis_weights.zip](https://drive.google.com/file/d/1muyIgrArhfqPDXv1NvHO7Fc1K01PFKH9/view?usp=sharing)

Архив содержит:
- `dirt_weights.pt` - Модель классификации чистоты (ConvNeXt Tiny)
- `best_yolo_damage.pt` - YOLO модель для детекции повреждений

**Установка весов:**
```bash
# Создайте папку weights (если не существует)
mkdir -p weights

# Распакуйте скачанный архив в папку weights/
unzip car_analysis_weights.zip -d weights/

# Проверьте наличие основных файлов
ls weights/
# Должны быть файлы: dirt_weights.pt, best_yolo_damage.pt
```

> 💡 **Примечание**: Без этих весов система не сможет выполнять анализ. Модели обучены на специализированном датасете автомобильных повреждений и состояний чистоты.

### 1. Использование Docker и Docker Compose (рекомендуется)

**После загрузки весов в папку weights/ запустите:**

```bash
sudo docker compose up -d --build 
```

Откройте в браузере: http://localhost:8000

### 2. Установка зависимостей

```bash
# Установка основных зависимостей через uv
uv sync

# Установка дополнительных веб-зависимостей
uv pip install -r requirements_web.txt

# Альтернативно, установка отдельных пакетов
uv add ultralytics fastapi uvicorn python-multipart pillow numpy
```

### 3. Запуск веб-приложения

```bash
# Активация окружения и запуск
uv run python app.py

# Или с активированным окружением
source .venv/bin/activate  # Linux/Mac
python app.py
```

Откройте в браузере: http://localhost:8000


### 4. Использование через веб-интерфейс

1. Загрузите изображение автомобиля
2. Нажмите "Анализировать автомобиль"
3. Получите результаты:
   - Статус повреждений с выделением областей
   - Уровень чистоты
   - Общие рекомендации

### 5. Проверка системы

```bash
# Комплексная проверка состояния системы
uv run python health_check.py
```

Этот скрипт проверит:
- Версию Python и зависимости
- Наличие обученных моделей
- Доступность CUDA
- Структуру проекта
- Базовую функциональность

## 🎓 Обучение моделей

### Подготовка датасета

```bash
# Подготовка данных из папок clean/dirty
uv run python scripts/prepare_dataset.py --input-dir ./dataset --output-dir ./dataset_prepared
```

Ожидаемая структура входных данных:
```
dataset/
├── images/
│   ├── clean/    # Изображения чистых автомобилей
│   └── dirty/    # Изображения грязных автомобилей
└── annotations/
    └── _annotations.coco.json
```

### Обучение модели чистоты

```bash
# Бинарная классификация: чистый (1) / грязный (0)
uv run python train_cleanliness.py \
    --data-root ./dataset_prepared \
    --backbone convnext_tiny \
    --img-size 384 \
    --batch-size 16 \
    --epochs 20 \
    --lr 3e-4
```

**Параметры:**
- `--backbone`: Архитектура модели (convnext_tiny, efficientnet_b2, resnet50)
- `--img-size`: Размер входного изображения
- `--batch-size`: Размер батча
- `--epochs`: Количество эпох
- `--lr`: Скорость обучения

### Обучение модели детекции повреждений

```bash
# Бинарная классификация: поврежденный (1) / целый (0)
uv run python train_damage_detection.py \
    --data-root ./dataset_prepared \
    --backbone convnext_tiny \
    --img-size 384 \
    --batch-size 16 \
    --epochs 20 \
    --lr 3e-4
```

### Результаты обучения

После обучения в папке `runs/TIMESTAMP/` создаются:
- `best.pt` - лучшая модель
- `history.json` - история обучения
- `test_report.json` - метрики на тестовой выборке
- `val_pr_curve.png` - PR-кривая
- `val_confusion.png` - матрица ошибок

## 🔍 Архитектура системы

### Веб-приложение (app.py)

**CleanlinessNet** - Бинарная классификация чистоты:
```python
class CleanlinessNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=False, dropout=0.1):
        self.backbone = timm.create_model(backbone, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1)  # Бинарная классификация
        )
```

**IntegratedCarAnalyzer** - Главный анализатор:
- Загружает обученные модели
- Комбинирует результаты YOLO и классификации чистоты
- Создает аннотированные изображения
- Формирует итоговые отчеты

### YOLO модель для повреждений

Поддерживаемые классы повреждений:
```python
DAMAGE_CLASSES = {
    0: "damaged door",       # Поврежденная дверь
    1: "damaged window",     # Поврежденное окно
    2: "damaged headlight",  # Поврежденная фара
    3: "damaged mirror",     # Поврежденное зеркало
    4: "dent",               # Вмятина
    5: "damaged hood",       # Поврежденный капот
    6: "damaged bumper",     # Поврежденный бампер
    7: "damaged wind shield" # Поврежденное лобовое стекло
}
```

### Модель классификации чистоты

- **Архитектура**: ConvNeXt Tiny / EfficientNet
- **Тип задачи**: Бинарная классификация
- **Классы**: 
  - 0 = Грязный автомобиль
  - 1 = Чистый автомобиль
- **Метрики**: Точность, полнота, F1-score, AUC-ROC

## 📊 API эндпоинты

### POST /analyze
Анализ изображения автомобиля

**Запрос:**
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@car_image.jpg"
```

**Ответ:**
```json
{
  "damage_analysis": {
    "is_damaged": true,
    "damage_confidence": 0.95,
    "detected_damages": [
      {
        "type": "dent",
        "confidence": 0.95,
        "bbox": {"x1": 100, "y1": 150, "x2": 200, "y2": 250}
      }
    ],
    "damage_count": 1
  },
  "cleanliness_analysis": {
    "clean_level": 0,
    "clean_probabilities": [0.8, 0.2],
    "cleanliness_status": {"status": "Грязный", "color": "#8B4513"},
    "clean_confidence": 0.2
  },
  "overall_status": {
    "status": "Требует ремонта",
    "color": "#DC143C",
    "priority": "high"
  },
  "annotated_image": "data:image/jpeg;base64,..."
}
```

### GET /health
Проверка состояния сервиса

### GET /
Главная страница с веб-интерфейсом

## 🔧 Конфигурация

### Пути к моделям

Система автоматически ищет модели в следующих местах:

**Модель чистоты:**
- `weights/dirt_weights.pt`

**YOLO модель для детекции повреждений:**
- `weights/best_yolo_damage.pt`

### Параметры обучения

Рекомендуемые параметры для разных задач:

**Быстрое обучение (тестирование):**
```bash
--backbone convnext_tiny --img-size 224 --batch-size 32 --epochs 5
```

**Качественное обучение:**
```bash
--backbone convnext_base --img-size 384 --batch-size 16 --epochs 50
```

**Максимальное качество:**
```bash
--backbone efficientnet_b4 --img-size 512 --batch-size 8 --epochs 100
```

## 🔍 Инференс и предсказания

### Использование predict.py

```bash
# Автоматическое определение модели и задачи
uv run python predict.py --checkpoint weights/dirt_weights.pt --images test_car.jpg

# Пакетная обработка с сохранением аннотаций
uv run python predict.py \
    --checkpoint runs/damage_exp/best.pt \
    --images ./test_images/ \
    --task damage \
    --threshold 0.7 \
    --save-individual \
    --batch-size 8

# Кастомный выходной каталог
uv run python predict.py \
    --checkpoint weights/dirt_weights.pt \
    --images ./cars/ \
    --out ./my_predictions/ \
    --max-samples 20
```

### Структура выходных данных predict.py

```
predictions_TIMESTAMP/
├── predictions.csv           # Детальные результаты в CSV
├── summary.png              # Сводная статистика и графики
├── sample_grid.png          # Сетка примеров предсказаний
├── metadata.json            # Метаданные эксперимента
└── individual/              # Аннотированные изображения (опционально)
    ├── car1_pred_clean.jpg
    ├── car2_pred_dirty.jpg
    └── ...
```

### Примеры использования

```bash
# Запуск интерактивных примеров
uv run python predict_examples.py
```

## �📈 Метрики и оценка

### Классификация чистоты
- **Accuracy**: Общая точность
- **Precision/Recall**: По каждому классу
- **F1-Score**: Гармоническое среднее
- **AUC-ROC**: Площадь под ROC-кривой
- **Confusion Matrix**: Матрица ошибок

### Детекция повреждений (YOLO)
- **mAP@0.5**: Средняя точность при IoU=0.5
- **mAP@0.5:0.95**: Средняя точность при IoU от 0.5 до 0.95
- **Precision/Recall**: По каждому классу повреждений

## � Производительность и оптимизация

### Рекомендации по обучению

**Выбор backbone:**
- `convnext_tiny`: Быстрое обучение, хорошее качество
- `convnext_base`: Баланс скорости и качества  
- `efficientnet_b2`: Высокое качество, умеренная скорость
- `efficientnet_b4`: Максимальное качество, медленнее

**Гиперпараметры по задачам:**
```bash
# Быстрый эксперимент
--epochs 5 --batch-size 32 --img-size 224

# Продакшн качество
--epochs 50 --batch-size 16 --img-size 384 --lr 2e-4

# Максимальное качество  
--epochs 100 --batch-size 8 --img-size 512 --lr 1e-4
```





### Ошибки GPU/CUDA
```bash
# Проверьте доступность CUDA
uv run python -c "import torch; print(torch.cuda.is_available())"

# Принудительно используйте CPU
export CUDA_VISIBLE_DEVICES=""

# Проверка версий PyTorch
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

## 📦 Модели и веса

### Предобученные модели

Система использует две основные модели:

1. **🧽 Модель классификации чистоты** (`dirt_weights.pt`)
   - **Архитектура**: ConvNeXt Tiny
   - **Задача**: Бинарная классификация (чистый/грязный)

2. **🔧 Модель детекции повреждений** (`best_yolo_damage.pt`)
   - **Архитектура**: YOLOv8n
   - **Задача**: Object detection для 8 типов повреждений
   - **Классы**: damaged_door, damaged_window, damaged_headlight, damaged_mirror, dent, damaged_hood, damaged_bumper, damaged_windshield


### Проблемы с памятью
```bash
# Уменьшение размера батча
--batch-size 4  # вместо 16

# Уменьшение размера изображения
--img-size 224  # вместо 384

# Использование градиентного накопления
--accumulate-grad-batches 4
```

### Медленное обучение
```bash
# Использование смешанной точности (если поддерживается)
--mixed-precision

# Оптимизация количества воркеров
--num-workers 4  # адаптируйте под ваш CPU

# Использование предобученных весов
--pretrained
```