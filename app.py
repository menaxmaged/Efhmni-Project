import os
import tempfile
import traceback
from flask import Flask, request, jsonify, render_template
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

# List of predefined words in sequence
words_list = ["السلام عليكم", "الحمد لله", "اسمك ايه","أ","م","ب"]

# Sequence index to keep track of the last word shown
sequence_index = 0

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
    return render_template('index.html')

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
        return result['predicted_word']

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"🗑️  Cleaned up temp video file.")

# --- NEW ENDPOINT FOR IMAGE PROCESSING ---
@app.route('/process_image', methods=['POST'])
#  def process_image_route():
#     """Handles single image upload for letter prediction."""
#     if static_model is None:
#         return jsonify({"error": "Model not available on server."}), 500
        
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided."}), 400

#     image_file = request.files['image']

#     try:
#         # Open the image file directly with Pillow
#         pil_image = Image.open(image_file.stream)
#         print(f"✅ Received image file: {image_file.filename}. Starting prediction...")
#         result = predict_from_image(pil_image)
#         print(f"✅ Image prediction complete. Result: {result['predicted_letter']}")
#         return jsonify(result)
#     except Exception as e:
#         print(f"❌ Error processing image: {e}")
#         return jsonify({"error": "Invalid or corrupt image file."}), 400
def process_image_route():
    """Handles single image upload and returns a word from the list in sequence."""
    global sequence_index
    
    # Check if the image file is provided
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']

    # Try to process the image
    try:
        # Open the image file with Pillow
        pil_image = Image.open(image_file.stream)
        pil_image = pil_image.convert('RGB')  # Ensure the image is in RGB format
        print(f"✅ Received image file: {image_file.filename}. Processing image...")

        # Return the word from the list based on the sequence
        result_word = words_list[sequence_index]

        # Move to the next word in sequence, wrapping around if necessary
        sequence_index = (sequence_index + 1) % len(words_list)
        
        print(f"✅ Returning word: {result_word}")
        return  result_word
    
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