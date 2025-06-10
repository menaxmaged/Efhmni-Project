import os
import tempfile
from flask import Flask, request, jsonify, render_template_string
from pyngrok import ngrok
from functions import video_to_frames, predict_from_frames # Assuming these are in functions.py

# --- Configuration ---
# It's better to load sensitive data like tokens from environment variables
# You would set this in your terminal: export NGROK_AUTHTOKEN='your_token'
NGROK_TOKEN = os.environ.get("NGROK_AUTHTOKEN")
# Fallback for easier testing if the environment variable isn't set
if not NGROK_TOKEN:
    NGROK_TOKEN = "2N1uyElcHqbXEsvE6616QFzSn4W_6rZ1Ek8vBJNGsKXyRhZ3P" # Your hardcoded token as a fallback

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# --- HTML Template ---
# A more user-friendly HTML page with some basic styling and JavaScript
# to show a loading message and display the results dynamically.
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
        input[type="file"] { border: 2px dashed #007bff; padding: 15px; border-radius: 5px; cursor: pointer; display: block; width: calc(100% - 34px); margin: 20px 0; }
        input[type="submit"] { background-color: #007bff; color: white; border: none; padding: 15px 30px; border-radius: 5px; font-size: 16px; cursor: pointer; transition: background-color 0.3s; }
        input[type="submit"]:hover { background-color: #0056b3; }
        #loader { display: none; margin: 20px auto; border: 5px solid #f3f3f3; border-top: 5px solid #007bff; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
        #result { margin-top: 20px; padding: 15px; background: #e9f5ff; border-left: 5px solid #007bff; text-align: right; font-size: 18px; font-weight: bold; word-wrap: break-word; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>مترجم لغة الإشارة المصرية</h1>
        <form id="uploadForm">
            <label for="video">اختر فيديو (.mp4):</label>
            <input type="file" id="video" name="video" accept="video/mp4" required>
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
            const fileInput = document.getElementById('video');
            const formData = new FormData();

            if (fileInput.files.length === 0) {
                resultDiv.textContent = 'الرجاء اختيار ملف فيديو أولاً.';
                resultDiv.style.color = 'red';
                return;
            }

            formData.append('video', fileInput.files[0]);
            resultDiv.textContent = '';
            loader.style.display = 'block';

            try {
                const response = await fetch('/process_video', {
                    method: 'POST',
                    body: formData
                });

                loader.style.display = 'none';

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'حدث خطأ غير متوقع.');
                }

                const data = await response.json();
                resultDiv.textContent = 'الجملة المترجمة: ' + data.predicted_sentence;
                resultDiv.style.color = '#333';

            } catch (error) {
                loader.style.display = 'none';
                resultDiv.textContent = 'خطأ: ' + error.message;
                resultDiv.style.color = 'red';
            }
        });
    </script>
</body>
</html>
"""

def start_ngrok(port: int):
    """Starts Ngrok tunnel and prints the public URL."""
    print(" * Starting ngrok tunnel...")
    if not NGROK_TOKEN:
        print(" * Warning: NGROK_AUTHTOKEN not set. Using hardcoded token.")
    ngrok.set_auth_token(NGROK_TOKEN)
    try:
        public_url = ngrok.connect(port, domain="tender-sculpin-badly.ngrok-free.app")
        print(f" * Ngrok tunnel is running at: {public_url}")
        return public_url
    except Exception as e:
        print(f" * Error starting ngrok: {e}")
        print(" * Please ensure your ngrok authtoken is correct and the domain is not already in use.")
        return None


@app.route("/")
def upload_form():
    """Serves the main HTML upload form."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    Handles video upload, processing, and prediction.
    Uses a temporary file to handle the upload securely.
    """
    if 'video' not in request.files or not request.files['video'].filename:
        return jsonify({"error": "No video file provided."}), 400

    video_file = request.files['video']

    # Use a temporary file to securely save the uploaded video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_video:
        video_file.save(temp_video.name)
        video_path = temp_video.name

        print(f"Video saved temporarily at: {video_path}")

        try:
            # 1. Extract frames
            print("Extracting frames...")
            frames = video_to_frames(video_path, frame_interval=5) # Process every 5th frame
            if not frames:
                return jsonify({"error": "Could not extract any frames from the video."}), 400
            print(f"Extracted {len(frames)} frames.")

            # 2. Predict from frames
            print("Processing frames with the model...")
            result = predict_from_frames(frames)
            print(f"Prediction result: {result}")
            
            # The result from predict_from_frames is assumed to be a dictionary.
            # We return it directly as a JSON response.
            return jsonify(result)

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/<path:path>')
def catch_all(path):
    return f'Page not found: {path}', 404

# --- Main Execution ---
if __name__ == "__main__":
    start_ngrok(5050)
    app.run(port=5050, host="0.0.0.0")

