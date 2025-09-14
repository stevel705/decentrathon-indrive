# Car Condition Baseline (Multitask, PyTorch)

Базовый пайплайн обучения и проверки модели для задач:
- **damage**: битый (1) / небитый (0)
- **cleanliness**: уровни чистоты (по умолчанию 3 класса: 0=clean, 1=slightly dirty, 2=strongly dirty)

Ожидается структура после препроцессинга:
```
data/_merged_coco/
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
```

## Установка
```bash
pip install torch torchvision timm pandas scikit-learn matplotlib
```

## Запуск обучения и валидации
```bash
python train_eval.py   --data-root /path/to/data/_merged_coco   --backbone convnext_tiny   --img-size 384   --batch-size 32   --epochs 10   --clean-levels 3
```

Результаты (чекпойнты, графики, отчёты) сохраняются в `runs/<timestamp>/`.

## Инференс на новых фото
```bash
python predict.py   --checkpoint runs/<timestamp>/best.pt   --backbone convnext_tiny   --img-size 384   --images /path/to/folder_or_image.jpg   --out runs/<timestamp>/pred_samples
```
Скрипт сохранит превью с предсказаниями и вероятностями.

