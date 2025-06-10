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
# 1. MODEL AND CONFIGURATION LOADING
# =============================================================================

# --- Model: Static Sign Recognition (Letters) ---
# We are only using the static model for letter-by-letter translation from video frames.
try:
    static_model = load_model('arabic_sign_language_model.h5', compile=False)
    print("✅ Static sign language model (arabic_sign_language_model.h5) loaded successfully.")
except Exception as e:
    print(f"❌ Error loading static model: {e}")
    print("🚨 Please ensure 'arabic_sign_language_model.h5' is in the same directory and is a valid Keras model file.")
    static_model = None

static_class_labels = [
    'ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain',
    'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra',
    'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw',
    'ya', 'yaa', 'zay'
]

# =============================================================================
# 2. HELPER FUNCTIONS (for prediction logic)
# =============================================================================

def preprocess_static_image(image):
    """Preprocesses an image for the static sign language model."""
    img = image.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_from_image(image):
    """Predicts a single static sign (letter) from an image."""
    # This function now relies on the check in the main route to ensure the model is loaded.
    processed_image = preprocess_static_image(image)
    predictions = static_model.predict(processed_image)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_label = static_class_labels[predicted_class_idx]

    return {
        'predicted_letter': predicted_label,
        'confidence': round(confidence, 4)
    }

def video_to_frames(video_path, frame_interval=1):
    """Extracts frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    if not cap.isOpened():
        # Raise an error if the video file is invalid or corrupted.
        raise IOError(f"Cannot open video file. It may be corrupt or in an unsupported format: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def predict_word_from_video(frames, confidence_threshold=0.6):
    """
    Predicts a sequence of static signs (letters) from video frames to form a word.
    
    Args:
        frames (list): A list of video frames (as NumPy arrays from OpenCV).
        confidence_threshold (float): The minimum confidence to consider a prediction.

    Returns:
        dict: A dictionary containing the final predicted word.
    """
    predicted_word = []
    last_prediction = None

    for i, frame in enumerate(frames):
        # Convert frame from OpenCV BGR format to PIL RGB format
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Use the function for single image prediction
        result = predict_from_image(pil_image)
        
        predicted_letter = result['predicted_letter']
        confidence = result['confidence']

        print(f"  > Frame {i+1}/{len(frames)} -> Prediction: {predicted_letter} (Confidence: {confidence:.2f})")

        # Add the letter if confidence is high and it's a different letter from the last
        if confidence > confidence_threshold and predicted_letter != last_prediction:
            predicted_word.append(predicted_letter)
            last_prediction = predicted_letter
    
    # Join the letters to form a word, separated by hyphens
    final_word = "-".join(predicted_word) if predicted_word else "Sorry, no letters were clearly recognized."
    return {"predicted_word": final_word}


# =============================================================================
# 3. FLASK APPLICATION
# =============================================================================

# --- Flask App Initialization and Ngrok Configuration ---
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
NGROK_TOKEN = os.environ.get("NGROK_AUTHTOKEN", "2N1uyElcHqbXEsvE6616QFzSn4W_6rZ1Ek8vBJNGsKXyRhZ3P")

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>مترجم لغة الإشارة المصرية</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; max-width: 500px; width: 100%; }
        h1 { color: #0056b3; }
        input[type="file"] { border: 2px dashed #007bff; padding: 15px; border-radius: 5px; cursor: pointer; display: block; width: calc(100% - 34px); margin: 10px 0; }
        input[type="submit"] { background-color: #007bff; color: white; border: none; padding: 15px 30px; border-radius: 5px; font-size: 16px; cursor: pointer; transition: background-color 0.3s; margin-top: 10px; }
        input[type="submit"]:hover { background-color: #0056b3; }
        #loader { display: none; margin: 20px auto; border: 5px solid #f3f3f3; border-top: 5px solid #007bff; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
        #result { margin-top: 20px; padding: 15px; background: #e9f5ff; border-left: 5px solid #007bff; text-align: right; font-size: 18px; font-weight: bold; word-wrap: break-word; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>مترجم لغة الإشارة المصرية</h1>
        <p>قم بتحميل فيديو لترجمته إلى تسلسل حروف.</p>
        <form id="uploadForm">
            <label for="video_file">اختر فيديو (.mp4):</label>
            <input type="file" id="video_file" name="file" accept="video/mp4" required>
            <input type="submit" value="ابدأ الترجمة">
        </form>
        <div id="loader"></div>
        <div id="result"></div>
    </div>
    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(event) {
            event.preventDefault();
            const resultDiv = document.getElementById('result');
            const loader = document.getElementById('loader');
            const fileInput = document.getElementById('video_file');
            const formData = new FormData();

            if (fileInput.files.length === 0) {
                resultDiv.textContent = 'الرجاء اختيار ملف فيديو أولاً.';
                resultDiv.style.color = 'red';
                return;
            }

            formData.append('file', fileInput.files[0]);
            resultDiv.textContent = '';
            loader.style.display = 'block';

            try {
                const response = await fetch('/process_video', { method: 'POST', body: formData });
                
                if (!response.ok) {
                    let errorMsg;
                    const contentType = response.headers.get("content-type");
                    if (contentType && contentType.indexOf("application/json") !== -1) {
                        const errorData = await response.json();
                        errorMsg = errorData.error || 'An unexpected server error occurred.';
                    } else {
                        errorMsg = `Server returned an unexpected response (Status: ${response.status}). Check the server console for more details.`;
                    }
                    throw new Error(errorMsg);
                }

                const data = await response.json();
                if (data.predicted_word) {
                    resultDiv.textContent = 'الكلمة المترجمة (تسلسل حروف): ' + data.predicted_word;
                }
                resultDiv.style.color = '#333';

            } catch (error) {
                resultDiv.textContent = 'خطأ: ' + error.message;
                resultDiv.style.color = 'red';
            } finally {
                loader.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

def start_ngrok(port: int):
    """Starts Ngrok tunnel and prints the public URL."""
    print(" * Starting ngrok tunnel...")
    ngrok.set_auth_token(NGROK_TOKEN)
    try:
        public_url = ngrok.connect(port, domain="tender-sculpin-badly.ngrok-free.app")
        print(f" * Ngrok tunnel is running at: {public_url}")
    except Exception as e:
        print(f" * Error starting ngrok: {e}")

# --- Flask Routes ---

@app.errorhandler(Exception)
def handle_exception(e):
    """Global handler for any exception, ensures a JSON response for all errors."""
    # Log the full exception traceback to the console for debugging
    print(f"--- UNHANDLED EXCEPTION: {e} ---")
    traceback.print_exc()
    print("---------------------------------")
    # Return a JSON response with a 500 status code
    return jsonify(error="An internal server error occurred. Please check the server logs for details."), 500

@app.route("/")
def upload_form():
    """Serves the main HTML upload form."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/process_video', methods=['POST'])
def process_video():
    """Handles video upload for static sign (letter-by-letter) prediction."""
    # --- 1. Pre-computation checks ---
    if static_model is None:
        print("❌ Prediction failed because the static model is not loaded.")
        return jsonify({"error": "Model not available on server. Please check server logs."}), 500
        
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({"error": "No video file provided."}), 400

    video_file = request.files['file']
    print(f"\n✅ Received video file: {video_file.filename}")

    # --- 2. File processing and prediction ---
    # The 'with' statement ensures the temporary file is always cleaned up
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        video_path = temp_video.name
        video_file.save(video_path)
    
    try:
        print(f"⚙️  Extracting frames from '{video_path}'...")
        frames = video_to_frames(video_path, frame_interval=5)
        if not frames:
            print("❌ No frames were extracted. The video file may be corrupt or empty.")
            return jsonify({"error": "Could not extract any frames from the video. The file may be corrupt or in an unsupported format."}), 400
        
        print(f"✅ Extracted {len(frames)} frames. Starting prediction...")
        result = predict_word_from_video(frames)
        print(f"✅ Prediction complete. Result: {result['predicted_word']}")
        
        return jsonify(result)

    finally:
        # --- 3. Cleanup ---
        # Ensure the temporary file is deleted after processing
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"🗑️  Cleaned up temporary file: {video_path}")


@app.route('/<path:path>')
def catch_all(path):
    """Catches all other routes and returns a JSON 404 error."""
    return jsonify({"error": f"Endpoint not found: /{path}"}), 404

# --- Main Execution ---
if __name__ == "__main__":
    if static_model is None:
        print("\n🚨 WARNING: The application is starting, but the model failed to load.")
        print("   The /process_video endpoint will return an error until the model is fixed.\n")
    start_ngrok(5050)
    app.run(port=5050, host="0.0.0.0")
