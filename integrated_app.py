import os
import glob
import torch
import torch.nn as nn
import numpy as np
import uvicorn
from typing import Dict, Any
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import io
import base64
from ultralytics import YOLO

try:
    import timm
except ImportError:
    timm = None

# Константы
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Классы повреждений из YOLO модели
DAMAGE_CLASSES = {
    0: "damaged door",
    1: "damaged window", 
    2: "damaged headlight",
    3: "damaged mirror",
    4: "dent",
    5: "damaged hood",
    6: "damaged bumper",
    7: "damaged wind shield"
}

class ConditionNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", clean_levels=2):
        super().__init__()
        if timm is None:
            raise ImportError("Please install timm: pip install timm")
        
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.damage_head = nn.Linear(feat_dim, 1)
        self.clean_head = nn.Linear(feat_dim, clean_levels if clean_levels > 2 else 1)
        self.clean_levels = clean_levels

    def forward(self, x):
        f = self.backbone(x)
        d = self.damage_head(f).squeeze(1)
        c = self.clean_head(f)
        if self.clean_levels == 2:
            c = c.squeeze(1)
        return d, c

class IntegratedCarAnalyzer:
    def __init__(self, condition_checkpoint_path: str, yolo_model_path: str, 
                 backbone: str = "convnext_tiny", img_size: int = 384, clean_levels: int = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.clean_levels = clean_levels
        
        # Загружаем модель состояния (чистота/грязность)
        print(f"Загружаем модель состояния: {condition_checkpoint_path}")
        condition_checkpoint = torch.load(condition_checkpoint_path, map_location=self.device, weights_only=False)
        
        # Определяем количество уровней чистоты из размера clean_head в checkpoint
        if "model" in condition_checkpoint:
            model_state = condition_checkpoint["model"]
        else:
            model_state = condition_checkpoint
            
        clean_head_shape = model_state["clean_head.weight"].shape[0]
        actual_clean_levels = clean_head_shape if clean_head_shape > 2 else 2
        
        # Создаем модель с правильными параметрами
        self.condition_model = ConditionNet(backbone=backbone, clean_levels=actual_clean_levels).to(self.device)
        self.condition_model.load_state_dict(model_state)
        self.condition_model.eval()
        
        # Обновляем self.clean_levels для корректной работы predict
        self.clean_levels = actual_clean_levels
        
        # Загружаем YOLO модель для детекции повреждений
        print(f"Загружаем YOLO модель: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Преобразования для изображений (для модели состояния)
        self.transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    def analyze_car(self, image: Image.Image) -> Dict[str, Any]:
        """Комплексный анализ автомобиля: повреждения + чистота"""
        
        # 1. Детекция повреждений с помощью YOLO
        damage_results = self._detect_damages(image)
        
        # 2. Анализ чистоты с помощью нашей модели
        cleanliness_results = self._analyze_cleanliness(image)
        
        # 3. Создание аннотированного изображения
        annotated_image = self._create_annotated_image(image, damage_results)
        
        # 4. Формирование итогового результата
        is_damaged = len(damage_results["detections"]) > 0
        
        return {
            "damage_analysis": {
                "is_damaged": is_damaged,
                "damage_confidence": damage_results["max_confidence"] if is_damaged else 0.0,
                "detected_damages": damage_results["detections"],
                "damage_count": len(damage_results["detections"])
            },
            "cleanliness_analysis": cleanliness_results,
            "overall_status": self._get_overall_status(is_damaged, cleanliness_results),
            "annotated_image": annotated_image
        }
    
    def _detect_damages(self, image: Image.Image) -> Dict[str, Any]:
        """Детекция повреждений с помощью YOLO"""
        # Конвертируем PIL в формат для YOLO
        image_array = np.array(image)
        
        # Запускаем детекцию
        results = self.yolo_model(image_array, verbose=False)
        
        detections = []
        max_confidence = 0.0
        
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    
                    if conf > 0.3:  # Порог уверенности
                        damage_type = DAMAGE_CLASSES.get(int(cls), f"damage_{int(cls)}")
                        
                        detections.append({
                            "type": damage_type,
                            "confidence": float(conf),
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1), 
                                "x2": float(x2),
                                "y2": float(y2)
                            }
                        })
                        
                        max_confidence = max(max_confidence, float(conf))
        
        return {
            "detections": detections,
            "max_confidence": max_confidence
        }
    
    def _analyze_cleanliness(self, image: Image.Image) -> Dict[str, Any]:
        """Анализ чистоты с помощью нашей модели"""
        # Подготавливаем изображение
        image_rgb = image.convert("RGB")
        x = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Делаем предсказание
        with torch.no_grad():
            damage_logits, clean_logits = self.condition_model(x)
            
            damage_prob = torch.sigmoid(damage_logits).cpu().item()
            
            if self.clean_levels == 3:
                clean_probs = torch.softmax(clean_logits, dim=1).cpu().numpy()[0]
                clean_level = int(np.argmax(clean_probs))
            else:
                clean_prob = torch.sigmoid(clean_logits).cpu().item()
                clean_probs = [1 - clean_prob, clean_prob]
                clean_level = 1 if clean_prob >= 0.5 else 0
        
        return {
            "general_damage_probability": float(damage_prob),
            "is_damaged_general": damage_prob >= 0.5,
            "clean_level": int(clean_level),
            "clean_probabilities": [float(p) for p in clean_probs],
            "cleanliness_status": self._get_cleanliness_status(clean_level)
        }
    
    def _create_annotated_image(self, image: Image.Image, damage_results: Dict[str, Any]) -> str:
        """Создает аннотированное изображение с выделенными повреждениями"""
        # Создаем копию изображения для аннотации
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Пытаемся загрузить шрифт
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except Exception:
                font = ImageFont.load_default()
        
        # Рисуем рамки и подписи для каждого повреждения
        colors = ["red", "orange", "yellow", "purple", "pink", "cyan", "magenta", "lime"]
        
        for i, detection in enumerate(damage_results["detections"]):
            bbox = detection["bbox"]
            damage_type = detection["type"]
            confidence = detection["confidence"]
            
            color = colors[i % len(colors)]
            
            # Рисуем рамку
            draw.rectangle(
                [(bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"])],
                outline=color,
                width=3
            )
            
            # Подпись
            label = f"{damage_type}: {confidence:.2f}"
            
            # Рисуем фон для текста
            text_bbox = draw.textbbox((bbox["x1"], bbox["y1"] - 30), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            
            # Рисуем текст
            draw.text((bbox["x1"], bbox["y1"] - 30), label, fill="white", font=font)
        
        # Конвертируем в base64 для передачи на фронтенд
        buffer = io.BytesIO()
        annotated.save(buffer, format="JPEG", quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
    
    def _get_cleanliness_status(self, clean_level: int) -> Dict[str, str]:
        """Определяет статус чистоты"""
        if self.clean_levels == 3:
            if clean_level == 0:
                return {"status": "Грязный", "color": "#8B4513"}
            elif clean_level == 1:
                return {"status": "Умеренно загрязненный", "color": "#FFA500"}
            else:
                return {"status": "Чистый", "color": "#008000"}
        else:
            if clean_level == 0:
                return {"status": "Грязный", "color": "#8B4513"}
            else:
                return {"status": "Чистый", "color": "#008000"}
    
    def _get_overall_status(self, is_damaged: bool, cleanliness_results: Dict[str, Any]) -> Dict[str, Any]:
        """Определяет общий статус автомобиля"""
        if is_damaged:
            status = "Требует ремонта"
            color = "#DC143C"
            priority = "high"
        elif cleanliness_results["clean_level"] == 0:  # Грязный
            status = "Требует мойки"
            color = "#FF8C00"
            priority = "medium"
        elif self.clean_levels == 3 and cleanliness_results["clean_level"] == 1:  # Умеренно загрязненный
            status = "Слегка загрязнен"
            color = "#FFD700"
            priority = "low"
        else:
            status = "В отличном состоянии"
            color = "#008000"
            priority = "none"
        
        return {
            "status": status,
            "color": color,
            "priority": priority
        }

# Инициализация приложения
app = FastAPI(title="Integrated Car Analyzer", description="Комплексный анализ состояния автомобиля")

# Создаем папку для статических файлов
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Инициализация анализатора
def find_model_files():
    """Находит файлы моделей"""
    # Ищем модель состояния
    condition_models = glob.glob("runs/**/best*.pt", recursive=True)
    condition_model = None
    
    for model_path in condition_models:
        if "best" in model_path and os.path.exists(model_path):
            condition_model = model_path
            break
    
    if not condition_model:
        raise FileNotFoundError("Не найдена модель состояния! Убедитесь, что модель обучена.")
    
    # Ищем YOLO модель
    yolo_model_path = "weights/best_yolo_damage.pt"
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"Не найдена YOLO модель по пути: {yolo_model_path}")
    
    return condition_model, yolo_model_path

try:
    condition_model_path, yolo_model_path = find_model_files()
    print(f"Используем модель состояния: {condition_model_path}")
    print(f"Используем YOLO модель: {yolo_model_path}")
    
    analyzer = IntegratedCarAnalyzer(condition_model_path, yolo_model_path)
    print("Анализатор успешно инициализирован!")
    
except Exception as e:
    print(f"Ошибка при инициализации: {e}")
    analyzer = None

@app.get("/", response_class=HTMLResponse)
async def main_page():
    """Главная страница с UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Анализатор состояния автомобиля</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background-color: #fafafa;
            }
            .upload-area:hover {
                border-color: #007bff;
                background-color: #f0f8ff;
            }
            .file-input {
                margin: 20px 0;
            }
            .analyze-btn {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
            }
            .analyze-btn:hover {
                background-color: #0056b3;
            }
            .analyze-btn:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
            .results {
                margin-top: 30px;
                display: none;
            }
            .result-section {
                margin: 20px 0;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #007bff;
                background-color: #f8f9fa;
            }
            .status-badge {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                color: white;
                font-weight: bold;
                margin: 5px;
            }
            .image-container {
                text-align: center;
                margin: 20px 0;
            }
            .preview-image, .result-image {
                max-width: 100%;
                max-height: 400px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .damage-list {
                list-style: none;
                padding: 0;
            }
            .damage-item {
                background: #fff;
                margin: 10px 0;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #dc3545;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007bff;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 2s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Анализатор состояния автомобиля</h1>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📁 Нажмите здесь или перетащите изображение автомобиля</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
            </div>
            
            <div class="image-container" id="previewContainer" style="display: none;">
                <h3>Предварительный просмотр:</h3>
                <img id="previewImage" class="preview-image">
            </div>
            
            <div style="text-align: center;">
                <button class="analyze-btn" id="analyzeBtn" onclick="analyzeImage()" disabled>
                    🔍 Анализировать автомобиль
                </button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Анализируем изображение...</p>
            </div>
            
            <div class="results" id="results">
                <h2>📊 Результаты анализа</h2>
                
                <div class="result-section">
                    <h3>🎯 Общий статус</h3>
                    <div id="overallStatus"></div>
                </div>
                
                <div class="result-section">
                    <h3>🔧 Анализ повреждений</h3>
                    <div id="damageAnalysis"></div>
                </div>
                
                <div class="result-section">
                    <h3>🧽 Анализ чистоты</h3>
                    <div id="cleanlinessAnalysis"></div>
                </div>
                
                <div class="image-container" id="resultImageContainer" style="display: none;">
                    <h3>📷 Изображение с выделенными повреждениями:</h3>
                    <img id="resultImage" class="result-image">
                </div>
            </div>
        </div>
        
        <script>
            let selectedFile = null;
            
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('previewImage').src = e.target.result;
                        document.getElementById('previewContainer').style.display = 'block';
                        document.getElementById('analyzeBtn').disabled = false;
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            async function analyzeImage() {
                if (!selectedFile) return;
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = true;
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        displayResults(result);
                    } else {
                        alert('Ошибка: ' + result.detail);
                    }
                } catch (error) {
                    alert('Ошибка сети: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                }
            }
            
            function displayResults(result) {
                // Общий статус
                const overallStatus = result.overall_status;
                document.getElementById('overallStatus').innerHTML = 
                    `<span class="status-badge" style="background-color: ${overallStatus.color}">
                        ${overallStatus.status}
                    </span>`;
                
                // Анализ повреждений
                const damageAnalysis = result.damage_analysis;
                let damageHtml = '';
                
                if (damageAnalysis.is_damaged) {
                    damageHtml += `
                        <p><strong>⚠️ Обнаружены повреждения!</strong></p>
                        <p>Количество повреждений: ${damageAnalysis.damage_count}</p>
                        <p>Максимальная уверенность: ${(damageAnalysis.damage_confidence * 100).toFixed(1)}%</p>
                        <ul class="damage-list">
                    `;
                    
                    damageAnalysis.detected_damages.forEach(damage => {
                        damageHtml += `
                            <li class="damage-item">
                                <strong>${damage.type}</strong><br>
                                Уверенность: ${(damage.confidence * 100).toFixed(1)}%
                            </li>
                        `;
                    });
                    
                    damageHtml += '</ul>';
                    
                    // Показываем аннотированное изображение
                    document.getElementById('resultImage').src = result.annotated_image;
                    document.getElementById('resultImageContainer').style.display = 'block';
                } else {
                    damageHtml = '<p>✅ Повреждения не обнаружены</p>';
                    document.getElementById('resultImageContainer').style.display = 'none';
                }
                
                document.getElementById('damageAnalysis').innerHTML = damageHtml;
                
                // Анализ чистоты
                const cleanlinessAnalysis = result.cleanliness_analysis;
                const cleanlinessStatus = cleanlinessAnalysis.cleanliness_status;
                
                let cleanlinessHtml = `
                    <span class="status-badge" style="background-color: ${cleanlinessStatus.color}">
                        ${cleanlinessStatus.status}
                    </span>
                    <p>Уровень чистоты: ${cleanlinessAnalysis.clean_level}</p>
                `;
                
                if (cleanlinessAnalysis.clean_probabilities.length === 3) {
                    cleanlinessHtml += `
                        <p>Вероятности:</p>
                        <ul>
                            <li>Грязный: ${(cleanlinessAnalysis.clean_probabilities[0] * 100).toFixed(1)}%</li>
                            <li>Умеренно загрязненный: ${(cleanlinessAnalysis.clean_probabilities[1] * 100).toFixed(1)}%</li>
                            <li>Чистый: ${(cleanlinessAnalysis.clean_probabilities[2] * 100).toFixed(1)}%</li>
                        </ul>
                    `;
                } else {
                    cleanlinessHtml += `
                        <p>Вероятности:</p>
                        <ul>
                            <li>Грязный: ${(cleanlinessAnalysis.clean_probabilities[0] * 100).toFixed(1)}%</li>
                            <li>Чистый: ${(cleanlinessAnalysis.clean_probabilities[1] * 100).toFixed(1)}%</li>
                        </ul>
                    `;
                }
                
                document.getElementById('cleanlinessAnalysis').innerHTML = cleanlinessHtml;
                
                document.getElementById('results').style.display = 'block';
            }
            
            // Drag and drop support
            const uploadArea = document.querySelector('.upload-area');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#007bff';
                uploadArea.style.backgroundColor = '#f0f8ff';
            });
            
            uploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#ccc';
                uploadArea.style.backgroundColor = '#fafafa';
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#ccc';
                uploadArea.style.backgroundColor = '#fafafa';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    document.getElementById('fileInput').files = files;
                    document.getElementById('fileInput').dispatchEvent(new Event('change'));
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/analyze")
async def analyze_car_condition(file: UploadFile = File(...)):
    """Комплексный анализ состояния автомобиля"""
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Анализатор не инициализирован")
    
    # Проверяем тип файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        # Загружаем изображение
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Анализируем
        results = analyzer.analyze_car(image)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе изображения: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "analyzer_ready": analyzer is not None,
        "models": {
            "condition_model": analyzer is not None,
            "yolo_model": analyzer is not None
        }
    }

if __name__ == "__main__":
    if analyzer is None:
        print("ВНИМАНИЕ: Анализатор не инициализирован. Проверьте наличие моделей.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
