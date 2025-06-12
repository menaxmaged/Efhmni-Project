import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2

# Load model
try:
    static_model = load_model('static_model/arabic_sign_language_model.h5', compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    static_model = None

# Labels
static_class_labels = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain', 'ha', 'haa', 'jeem',
                       'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 'taa',
                       'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']

arabic_map = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'أ', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ', 'dhad': 'ض', 'fa': 'ف', 'gaaf': 'ق',
    'ghain': 'غ', 'ha': 'ه', 'haa': 'ح', 'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 'laam': 'ل', 'meem': 'م',
    'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'ta': 'ط', 'taa': 'ت', 'thaa': 'ث', 'thal': 'ذ',
    'toot': 'ة', 'waw': 'و', 'ya': 'ى', 'yaa': 'ي', 'zay': 'ز'
}

def preprocess_static_image(image):
    img = image.resize((224, 224)).convert('RGB')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_from_image(pil_image):
    if static_model is None:
        raise ValueError("Model not loaded.")
    processed_image = preprocess_static_image(pil_image)
    predictions = static_model.predict(processed_image)
    idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    label_en = static_class_labels[idx]
    label_ar = arabic_map.get(label_en, '?')
    return {'predicted_letter': label_ar, 'confidence': round(confidence, 4)}

def video_to_frames(video_path, frame_interval=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def predict_word_from_video_frames(frames, threshold=0.6):
    results = []
    last = None
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = predict_from_image(image)
        if result['confidence'] > threshold and result['predicted_letter'] != last:
            results.append(result['predicted_letter'])
            last = result['predicted_letter']
    return {"predicted_word": "".join(results) or "لم يتم التعرف على أية حروف بوضوح."}
