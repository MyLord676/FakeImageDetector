import os
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, send_file
from PIL import Image
from flask_cors import CORS
import yaml
from model import CFLNet
import timm

app = Flask(__name__)
CORS(app)

# Загрузка конфигурации и модели
CONFIG_PATH = "config/config.yaml"
MODEL_WEIGHTS_PATH = "checkpoints/best_model_auc.pth"   # путь к лучшей модели
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Читаем конфиг
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Определяем количество входных каналов для ASPP (как в trainer.py)
with torch.no_grad():
    test_model = timm.create_model(cfg['model_params']['encoder'], pretrained=False, features_only=True, out_indices=[4])
    in_planes = test_model(torch.randn(2, 3, 128, 128))[0].shape[1]
    del test_model

# Создаём модель и загружаем веса
model = CFLNet(cfg, in_planes).to(DEVICE)
state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(state_dict)
model.eval()   # режим инференса

IMG_SIZE = cfg['dataset_params']['im_size']   # обычно 256
MEAN = cfg['dataset_params']['mean']
STD = cfg['dataset_params']['std']

# Функция предобработки
def preprocess_image(image_bytes, target_size=(IMG_SIZE, IMG_SIZE)):
    # Открываем изображение, конвертируем в RGB и ресайзим
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0   # [0,1]
    # Нормализация по каналам (mean/std из конфига)
    for i in range(3):
        img_array[:,:,i] = (img_array[:,:,i] - MEAN[i]) / STD[i]
    # Превращаем в тензор (C, H, W) и добавляем batch
    img_tensor = torch.from_numpy(img_array).permute(2,0,1).unsqueeze(0).float()
    return img_tensor.to(DEVICE)

# Постобработка маски
def postprocess_mask(pred_tensor, original_size):
    """
    pred_tensor: torch.Tensor формы (1, num_class, H, W) — логиты
    original_size: (width, height) оригинала
    Возвращает байты PNG-маски (0 и 255)
    """
    with torch.no_grad():
        # Приводим к размеру исходного изображения (биланейная интерполяция)
        pred_up = F.interpolate(pred_tensor, size=(original_size[1], original_size[0]), 
                                mode='bilinear', align_corners=True)
        # Берём argmax по каналам (0 - фон, 1 - подделка)
        mask = torch.argmax(pred_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    # Превращаем в бинарную маску (0 или 255)
    mask_bin = (mask * 255).astype(np.uint8)
    # Кодируем в PNG
    _, buffer = cv2.imencode('.png', mask_bin)
    return buffer.tobytes()

# API-эндпоинт
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return {'error': 'No image file provided'}, 400
    file = request.files['image']
    if file.filename == '':
        return {'error': 'Empty filename'}, 400

    try:
        image_bytes = file.read()
        # Сохраняем оригинальный размер (ширина, высота)
        pil_img = Image.open(io.BytesIO(image_bytes))
        original_size = pil_img.size   # (width, height)

        # Предобработка
        input_tensor = preprocess_image(image_bytes)

        # Инференс
        with torch.no_grad():
            pred_logits, _ = model(input_tensor)   # (1, 2, H, W)

        # Постобработка
        mask_bytes = postprocess_mask(pred_logits, original_size)

        # Отправляем маску как PNG
        return send_file(
            io.BytesIO(mask_bytes),
            mimetype='image/png',
            as_attachment=False,
            download_name='mask.png'
        )
    except Exception as e:
        return {'error': str(e)}, 500

# Запуск
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=cfg['global_params']['port'], debug=True)