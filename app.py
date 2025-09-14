import io
import os
import glob
from typing import Dict, Any

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

try:
    import timm
except ImportError:
    timm = None

# Константы
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

class CarConditionPredictor:
    def __init__(self, checkpoint_path: str, backbone: str = "convnext_tiny", 
                 img_size: int = 384, clean_levels: int = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.clean_levels = clean_levels
        
        # Загружаем модель
        # Сначала загружаем checkpoint чтобы определить правильное количество clean_levels
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Определяем количество уровней чистоты из размера clean_head в checkpoint
        clean_head_shape = checkpoint["model"]["clean_head.weight"].shape[0]
        actual_clean_levels = clean_head_shape if clean_head_shape > 2 else 2
        
        # Создаем модель с правильными параметрами
        self.model = ConditionNet(backbone=backbone, clean_levels=actual_clean_levels).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        
        # Обновляем self.clean_levels для корректной работы predict
        self.clean_levels = actual_clean_levels
        
        # Преобразования для изображений
        self.transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Предсказывает состояние автомобиля по изображению"""
        # Подготавливаем изображение
        image_rgb = image.convert("RGB")
        x = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Делаем предсказание
        with torch.no_grad():
            damage_logits, clean_logits = self.model(x)
            
            # Вероятность повреждения
            damage_prob = torch.sigmoid(damage_logits).item()
            
            # Уровень чистоты
            if self.clean_levels == 2:
                clean_prob = torch.sigmoid(clean_logits).item()
                clean_level = 1 if clean_prob >= 0.5 else 0
                clean_probs = [1 - clean_prob, clean_prob]
            else:
                clean_probs = torch.softmax(clean_logits, dim=1).cpu().numpy()[0]
                clean_level = clean_probs.argmax()
        
        return {
            "damage_probability": float(damage_prob),
            "is_damaged": damage_prob >= 0.5,
            "clean_level": int(clean_level),
            "clean_probabilities": [float(p) for p in clean_probs],
            "status": self._get_status(damage_prob, clean_level)
        }
    
    def _get_status(self, damage_prob: float, clean_level: int) -> Dict[str, Any]:
        """Определяет статус и цвет для отображения"""
        # Определяем статус повреждения
        if damage_prob >= 0.7:
            damage_status = {"text": "Сильно поврежден", "color": "red", "severity": "high"}
        elif damage_prob >= 0.5:
            damage_status = {"text": "Поврежден", "color": "orange", "severity": "medium"}
        else:
            damage_status = {"text": "Не поврежден", "color": "green", "severity": "low"}
        
        # Определяем статус чистоты (для 3 уровней: 0=грязный, 1=средний, 2=чистый)
        if self.clean_levels == 3:
            clean_statuses = [
                {"text": "Грязный", "color": "red", "level": 0},
                {"text": "Средней чистоты", "color": "yellow", "level": 1},
                {"text": "Чистый", "color": "green", "level": 2}
            ]
            clean_status = clean_statuses[clean_level]
        else:
            # Для 2 уровней: 0=грязный, 1=чистый
            clean_status = {
                "text": "Чистый" if clean_level == 1 else "Грязный",
                "color": "green" if clean_level == 1 else "red",
                "level": clean_level
            }
        
        return {
            "damage": damage_status,
            "cleanliness": clean_status
        }

# Инициализация приложения
app = FastAPI(title="Car Condition Analyzer", description="Анализ состояния автомобиля по фотографии")

# Инициализация модели (используем последнюю обученную модель)
checkpoint_path = "runs/best/best.pt"
if not os.path.exists(checkpoint_path):
    # Попробуем найти любую доступную модель
    models = glob.glob("runs/**/best.pt", recursive=True)
    if models:
        checkpoint_path = models[0]
        print(f"Используем модель: {checkpoint_path}")
    else:
        raise FileNotFoundError("Не найдена обученная модель! Убедитесь, что модель обучена.")

predictor = CarConditionPredictor(checkpoint_path)

@app.get("/", response_class=HTMLResponse)
async def main_page():
    """Главная страница с UI"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Анализ состояния автомобиля</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
                grid-column: 1 / -1;
            }
            .main-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                min-height: 600px;
            }
            .left-panel {
                background: #fafafa;
                border-radius: 10px;
                padding: 20px;
                border: 2px solid #e9ecef;
            }
            .right-panel {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                border: 2px solid #e9ecef;
            }
            @media (max-width: 768px) {
                .main-content {
                    grid-template-columns: 1fr;
                    gap: 20px;
                }
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 30px;
                text-align: center;
                margin-bottom: 20px;
                background-color: white;
                transition: border-color 0.3s;
                min-height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }
            .upload-area:hover {
                border-color: #007bff;
            }
            .upload-area.dragover {
                border-color: #007bff;
                background-color: #e3f2fd;
            }
            #file-input {
                display: none;
            }
            .upload-btn {
                background-color: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
            }
            .upload-btn:hover {
                background-color: #0056b3;
            }
            .analyze-btn {
                background-color: #28a745;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
                display: none;
                width: 100%;
                max-width: 300px;
            }
            .analyze-btn:hover {
                background-color: #218838;
            }
            .analyze-btn:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
            }
            .preview-container {
                display: none;
                margin: 20px 0;
                text-align: center;
            }
            .preview-image {
                max-width: 100%;
                max-height: 300px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 15px;
            }
            .results {
                display: none;
            }
            .results h2 {
                color: #333;
                margin-bottom: 20px;
                text-align: center;
            }
            .result-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 15px 0;
                border-left: 5px solid #ccc;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .result-card.green {
                border-left-color: #28a745;
                background-color: #d4edda;
            }
            .result-card.yellow {
                border-left-color: #ffc107;
                background-color: #fff3cd;
            }
            .result-card.orange {
                border-left-color: #fd7e14;
                background-color: #ffeaa7;
            }
            .result-card.red {
                border-left-color: #dc3545;
                background-color: #f8d7da;
            }
            .result-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .result-value {
                font-size: 16px;
                margin-bottom: 8px;
            }
            .probability-bar {
                background-color: #e9ecef;
                border-radius: 10px;
                height: 20px;
                margin: 10px 0;
                overflow: hidden;
            }
            .probability-fill {
                height: 100%;
                transition: width 0.5s ease;
                border-radius: 10px;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
                grid-column: 1 / -1;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007bff;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                color: #dc3545;
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 5px;
                padding: 15px;
                margin: 10px 0;
                display: none;
                grid-column: 1 / -1;
            }
            .panel-title {
                font-size: 20px;
                font-weight: bold;
                color: #333;
                margin-bottom: 20px;
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 2px solid #e9ecef;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Анализ состояния автомобиля</h1>
            
            <div class="error" id="error-message"></div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Анализируем изображение...</p>
            </div>
            
            <div class="main-content">
                <!-- Левая панель - Загрузка изображения -->
                <div class="left-panel">
                    <div class="panel-title">📸 Загрузка изображения</div>
                    
                    <div class="upload-area" id="upload-area">
                        <p>Перетащите изображение автомобиля сюда или</p>
                        <button class="upload-btn" onclick="document.getElementById('file-input').click()">
                            Выберите файл
                        </button>
                        <input type="file" id="file-input" accept="image/*" onchange="handleFileSelect(event)">
                        <p style="font-size: 14px; color: #666; margin-top: 10px;">
                            Поддерживаемые форматы: JPG, PNG, BMP
                        </p>
                    </div>
                    
                    <div class="preview-container" id="preview-container">
                        <img id="preview-image" class="preview-image" alt="Предварительный просмотр">
                        <div>
                            <button class="analyze-btn" id="analyze-btn" onclick="analyzeImage()">
                                Анализировать изображение
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Правая панель - Результаты анализа -->
                <div class="right-panel">
                    <div class="panel-title">📊 Результаты анализа</div>
                    
                    <div class="results" id="results">
                        <div class="result-card" id="damage-card">
                            <div class="result-title">🔧 Повреждения</div>
                            <div class="result-value" id="damage-status"></div>
                            <div class="result-value">Вероятность повреждения: <span id="damage-probability"></span></div>
                            <div class="probability-bar">
                                <div class="probability-fill" id="damage-bar"></div>
                            </div>
                        </div>
                        
                        <div class="result-card" id="clean-card">
                            <div class="result-title">🧽 Чистота</div>
                            <div class="result-value" id="clean-status"></div>
                            <div class="result-value">Уровень чистоты: <span id="clean-level"></span></div>
                            <div id="clean-probabilities"></div>
                        </div>
                    </div>
                    
                    <div id="no-results" style="text-align: center; color: #666; margin-top: 50px;">
                        <p style="font-size: 18px;">📋</p>
                        <p>Загрузите изображение автомобиля для анализа</p>
                        <p style="font-size: 14px;">Результаты появятся здесь после обработки</p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let selectedFile = null;

            // Обработка drag & drop
            const uploadArea = document.getElementById('upload-area');
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, highlight, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, unhighlight, false);
            });

            function highlight(e) {
                uploadArea.classList.add('dragover');
            }

            function unhighlight(e) {
                uploadArea.classList.remove('dragover');
            }

            uploadArea.addEventListener('drop', handleDrop, false);

            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                handleFiles(files);
            }

            function handleFileSelect(event) {
                const files = event.target.files;
                handleFiles(files);
            }

            function handleFiles(files) {
                if (files.length === 0) return;
                
                const file = files[0];
                if (!file.type.startsWith('image/')) {
                    showError('Пожалуйста, выберите изображение');
                    return;
                }

                selectedFile = file;
                showPreview(file);
                hideError();
            }

            function showPreview(file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview-image');
                    preview.src = e.target.result;
                    document.getElementById('preview-container').style.display = 'block';
                    document.getElementById('analyze-btn').style.display = 'block';
                    document.getElementById('results').style.display = 'none';
                    document.getElementById('no-results').style.display = 'block';
                };
                reader.readAsDataURL(file);
            }

            async function analyzeImage() {
                if (!selectedFile) {
                    showError('Сначала выберите изображение');
                    return;
                }

                const analyzeBtn = document.getElementById('analyze-btn');
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = 'Анализируем...';
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                hideError();

                try {
                    const formData = new FormData();
                    formData.append('file', selectedFile);

                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();
                    displayResults(result);
                    
                } catch (error) {
                    console.error('Error:', error);
                    showError('Ошибка при анализе изображения: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = 'Анализировать изображение';
                }
            }

            function displayResults(result) {
                // Скрываем заглушку и показываем результаты
                document.getElementById('no-results').style.display = 'none';
                document.getElementById('results').style.display = 'block';

                // Отображение результатов по повреждениям
                const damageCard = document.getElementById('damage-card');
                const damageStatus = result.status.damage;
                damageCard.className = 'result-card ' + damageStatus.color;
                document.getElementById('damage-status').textContent = damageStatus.text;
                document.getElementById('damage-probability').textContent = 
                    (result.damage_probability * 100).toFixed(1) + '%';
                
                const damageBar = document.getElementById('damage-bar');
                damageBar.style.width = (result.damage_probability * 100) + '%';
                damageBar.style.backgroundColor = getBarColor(damageStatus.color);

                // Отображение результатов по чистоте
                const cleanCard = document.getElementById('clean-card');
                const cleanStatus = result.status.cleanliness;
                cleanCard.className = 'result-card ' + cleanStatus.color;
                document.getElementById('clean-status').textContent = cleanStatus.text;
                document.getElementById('clean-level').textContent = cleanStatus.level;

                // Отображение вероятностей по уровням чистоты
                const cleanProbsDiv = document.getElementById('clean-probabilities');
                const labels = ['Грязный', 'Средней чистоты', 'Чистый'];
                let probsHtml = '';
                
                result.clean_probabilities.forEach((prob, index) => {
                    if (index < labels.length) {
                        const percentage = (prob * 100).toFixed(1);
                        probsHtml += `
                            <div style="margin: 8px 0;">
                                <span style="font-size: 14px;">${labels[index]}: ${percentage}%</span>
                                <div class="probability-bar" style="height: 12px; margin-top: 3px;">
                                    <div class="probability-fill" 
                                         style="width: ${percentage}%; background-color: ${getLevelColor(index)};"></div>
                                </div>
                            </div>
                        `;
                    }
                });
                cleanProbsDiv.innerHTML = probsHtml;
            }

            function getBarColor(colorName) {
                const colors = {
                    'green': '#28a745',
                    'yellow': '#ffc107',
                    'orange': '#fd7e14',
                    'red': '#dc3545'
                };
                return colors[colorName] || '#6c757d';
            }

            function getLevelColor(level) {
                const colors = ['#dc3545', '#ffc107', '#28a745'];
                return colors[level] || '#6c757d';
            }

            function showError(message) {
                const errorDiv = document.getElementById('error-message');
                errorDiv.textContent = message;
                errorDiv.style.display = 'block';
            }

            function hideError() {
                document.getElementById('error-message').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_car_condition(file: UploadFile = File(...)):
    """API endpoint для анализа изображения автомобиля"""
    
    # Проверяем тип файла
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        # Читаем изображение
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Делаем предсказание
        result = predictor.predict(image)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
