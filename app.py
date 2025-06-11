import os
import tempfile
import traceback
from flask import Flask, request, jsonify, render_template_string
from pyngrok import ngrok
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# =============================================================================
# 0. MIDDLEWARE FOR URL NORMALIZATION
# =============================================================================
class SlashFixerMiddleware:
    """
    Server-side workaround for clients that might incorrectly generate URLs
    with leading slashes, like '//process_video'.
    """
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        environ['PATH_INFO'] = '/' + path.lstrip('/')
        return self.app(environ, start_response)

# =============================================================================
# 1. MODEL AND CONFIGURATION LOADING
# =============================================================================
try:
    static_model = load_model('arabic_sign_language_model.h5', compile=False)
    print("✅ Static sign language model (arabic_sign_language_model.h5) loaded successfully.")
except Exception as e:
    print(f"❌ Error loading static model: {e}")
    static_model = None

static_class_labels = [
    'ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain',
    'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra',
    'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw',
    'ya', 'yaa', 'zay'
]

arabic_map = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'أ', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ',
    'dhad': 'ض', 'fa': 'ف', 'gaaf': 'ق', 'ghain': 'غ', 'ha': 'ه', 'haa': 'ح',
    'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 'laam': 'ل', 'meem': 'م',
    'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'ta': 'ط',
    'taa': 'ت', 'thaa': 'ث', 'thal': 'ذ', 'toot': 'ة', 'waw': 'و', 'ya': 'ى',
    'yaa': 'ي', 'zay': 'ز'
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def preprocess_static_image(image):
    """Preprocesses a PIL image for the model."""
    img = image.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_from_image(pil_image):
    """Predicts a single static sign (letter) from a PIL image."""
    processed_image = preprocess_static_image(pil_image)
    predictions = static_model.predict(processed_image)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_label_en = static_class_labels[predicted_class_idx]
    predicted_label_ar = arabic_map.get(predicted_label_en, '?')

    return {
        'predicted_letter': predicted_label_ar,
        'confidence': round(confidence, 4)
    }

def video_to_frames(video_path, frame_interval=5):
    """Extracts frames from a video file path."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def predict_word_from_video_frames(frames, confidence_threshold=0.6):
    """Predicts a word from a list of video frames."""
    predicted_word_list = []
    last_prediction = None

    for i, frame in enumerate(frames):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = predict_from_image(pil_image)
        
        predicted_letter = result['predicted_letter']
        confidence = result['confidence']

        print(f"  > Frame {i+1}/{len(frames)} -> Prediction: {predicted_letter} (Confidence: {confidence:.2f})")

        if confidence > confidence_threshold and predicted_letter != last_prediction:
            predicted_word_list.append(predicted_letter)
            last_prediction = predicted_letter
    
    final_word = "".join(predicted_word_list) if predicted_word_list else "لم يتم التعرف على أية حروف بوضوح."
    return {"predicted_word": final_word}

# =============================================================================
# 3. FLASK APPLICATION
# =============================================================================

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
NGROK_TOKEN = os.environ.get("NGROK_AUTHTOKEN", "2N1uyElcHqbXEsvE6616QFzSn4W_6rZ1Ek8vBJNGsKXyRhZ3P") # Replace with your token

# --- Updated HTML Template for both Video and Image ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>مترجم لغة الإشارة المصرية</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background: white; padding: 20px 40px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; max-width: 600px; width: 100%; }
        h1 { color: #0056b3; margin-bottom: 20px; }
        .upload-section { margin-bottom: 30px; border-bottom: 1px solid #ddd; padding-bottom: 20px; }
        input[type="file"] { border: 2px dashed #007bff; padding: 15px; border-radius: 5px; cursor: pointer; display: block; width: calc(100% - 34px); margin: 10px 0; }
        input[type="submit"] { background-color: #007bff; color: white; border: none; padding: 12px 25px; border-radius: 5px; font-size: 16px; cursor: pointer; transition: background-color 0.3s; margin-top: 10px; }
        input[type="submit"]:hover { background-color: #0056b3; }
        #loader { display: none; margin: 20px auto; border: 5px solid #f3f3f3; border-top: 5px solid #007bff; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
        .result-box { margin-top: 20px; padding: 15px; background: #e9f5ff; border-left: 5px solid #007bff; text-align: right; font-size: 18px; font-weight: bold; word-wrap: break-word; min-height: 25px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>مترجم لغة الإشارة المصرية</h1>

        <div class="upload-section">
            <h2>ترجمة فيديو إلى كلمة</h2>
            <p>قم بتحميل فيديو (.mp4) لترجمته إلى تسلسل حروف.</p>
            <form id="videoUploadForm">
                <input type="file" id="video_file" name="video" accept="video/mp4" required>
                <input type="submit" value="ترجمة الفيديو">
            </form>
            <div id="videoLoader" class="loader" style="display:none;"></div>
            <div id="videoResult" class="result-box"></div>
        </div>

        <div class="upload-section">
            <h2>ترجمة صورة إلى حرف</h2>
            <p>قم بتحميل صورة (.jpg, .png) لترجمتها إلى حرف واحد.</p>
            <form id="imageUploadForm">
                <input type="file" id="image_file" name="image" accept="image/jpeg,image/png" required>
                <input type="submit" value="ترجمة الصورة">
            </form>
            <div id="imageLoader" class="loader" style="display:none;"></div>
            <div id="imageResult" class="result-box"></div>
        </div>
    </div>

    <script>
        // Generic function to handle form submission
        async function handleFormSubmit(event, formId, fileInputId, loaderId, resultId, endpoint) {
            event.preventDefault();
            const form = document.getElementById(formId);
            const fileInput = document.getElementById(fileInputId);
            const loader = document.getElementById(loaderId);
            const resultDiv = document.getElementById(resultId);
            const formData = new FormData(form);

            if (fileInput.files.length === 0) {
                resultDiv.textContent = 'الرجاء اختيار ملف أولاً.';
                resultDiv.style.color = 'red';
                return;
            }

            resultDiv.textContent = '';
            loader.style.display = 'block';

            try {
                const response = await fetch(endpoint, { method: 'POST', body: formData });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'حدث خطأ غير متوقع.');
                }

                if (data.predicted_word) {
                    resultDiv.textContent = 'الكلمة المترجمة: ' + data.predicted_word;
                } else if (data.predicted_letter) {
                    resultDiv.textContent = `الحرف المترجم: ${data.predicted_letter} (الدقة: ${(data.confidence * 100).toFixed(2)}%)`;
                }
                resultDiv.style.color = '#333';

            } catch (error) {
                resultDiv.textContent = 'خطأ: ' + error.message;
                resultDiv.style.color = 'red';
            } finally {
                loader.style.display = 'none';
            }
        }
        
        // Add event listeners for both forms
        document.getElementById('videoUploadForm').addEventListener('submit', (e) => handleFormSubmit(e, 'videoUploadForm', 'video_file', 'videoLoader', 'videoResult', '/process_video'));
        document.getElementById('imageUploadForm').addEventListener('submit', (e) => handleFormSubmit(e, 'imageUploadForm', 'image_file', 'imageLoader', 'imageResult', '/process_image'));
    </script>
</body>
</html>
"""

def start_ngrok(port: int):
    """Starts Ngrok tunnel and prints the public URL."""
    print(" * Starting ngrok tunnel...")
    ngrok.set_auth_token(NGROK_TOKEN)
    try:
        public_url = ngrok.connect(port,domain="tender-sculpin-badly.ngrok-free.app")
        print(f" * Ngrok tunnel is running at: {public_url}")
    except Exception as e:
        print(f" * Error starting ngrok: {e}")

# --- Flask Routes ---

@app.errorhandler(Exception)
def handle_exception(e):
    """Global handler for any exception."""
    traceback.print_exc()
    return jsonify(error="An internal server error occurred."), 500

@app.route("/")
def upload_form():
    """Serves the main HTML upload form."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/process_video', methods=['POST'])
def process_video_route():
    """Handles video upload for letter-by-letter prediction."""
    if static_model is None:
        return jsonify({"error": "Model not available on server."}), 500
        
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    video_file = request.files['video']
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        video_path = temp_video.name
        video_file.save(video_path)
    
    try:
        print(f"⚙️  Extracting frames from video...")
        frames = video_to_frames(video_path)
        if not frames:
            return jsonify({"error": "Could not extract frames from video."}), 400
        
        print(f"✅ Extracted {len(frames)} frames. Starting prediction...")
        result = predict_word_from_video_frames(frames)
        print(f"✅ Video prediction complete. Result: {result['predicted_word']}")
        return jsonify(result)

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"🗑️  Cleaned up temp video file.")

# --- NEW ENDPOINT FOR IMAGE PROCESSING ---
@app.route('/process_image', methods=['POST'])
def process_image_route():
    """Handles single image upload for letter prediction."""
    if static_model is None:
        return jsonify({"error": "Model not available on server."}), 500
        
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']

    try:
        # Open the image file directly with Pillow
        pil_image = Image.open(image_file.stream)
        print(f"✅ Received image file: {image_file.filename}. Starting prediction...")
        result = predict_from_image(pil_image)
        print(f"✅ Image prediction complete. Result: {result['predicted_letter']}")
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return jsonify({"error": "Invalid or corrupt image file."}), 400


@app.route('/<path:path>')
def catch_all(path):
    """Catches all other routes and returns a JSON 404 error."""
    return jsonify({"error": f"Endpoint not found: /{path}"}), 404

# --- Main Execution ---
if __name__ == "__main__":
    if static_model is None:
        print("\n🚨 WARNING: Application starting, but MODEL FAILED TO LOAD.")
    
    app.wsgi_app = SlashFixerMiddleware(app.wsgi_app)
    
    start_ngrok(5050)
    app.run(port=5050, host="0.0.0.0")