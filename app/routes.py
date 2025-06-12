import os, tempfile, traceback
from flask import request, jsonify, render_template
from pyngrok import ngrok
from .model_utils import static_model, predict_from_image, video_to_frames, predict_word_from_video_frames

NGROK_TOKEN = os.environ.get("NGROK_AUTHTOKEN", "2N1uyElcHqbXEsvE6616QFzSn4W_6rZ1Ek8vBJNGsKXyRhZ3P")

def start_ngrok(port: int):
    ngrok.set_auth_token(NGROK_TOKEN)
    try:
        public_url = ngrok.connect(port, domain="tender-sculpin-badly.ngrok-free.app")
        print(f" * Ngrok tunnel is running at: {public_url}")
    except Exception as e:
        print(f" * Error starting ngrok: {e}")

def register_routes(app):
    @app.errorhandler(Exception)
    def handle_exception(e):
        traceback.print_exc()
        return jsonify(error="An internal server error occurred."), 500

    @app.route("/")
    def upload_form():
        return render_template("index.html")

    @app.route("/process_image", methods=["POST"])
    def process_image_route():
        if static_model is None:
            return jsonify({"error": "Model not available."}), 500
        if 'image' not in request.files:
            return jsonify({"error": "No image provided."}), 400
        try:
            from PIL import Image
            image_file = request.files['image']
            image = Image.open(image_file.stream)
            result = predict_from_image(image)
            return jsonify(result)
        except Exception as e:
            print(f"❌ Image processing error: {e}")
            return jsonify({"error": "Invalid or corrupt image file."}), 400

    @app.route("/process_videoss", methods=["POST"])
    def process_video_route():
        if static_model is None:
            return jsonify({"error": "Model not available."}), 500
        if 'video' not in request.files:
            return jsonify({"error": "No video provided."}), 400

        video_file = request.files['video']
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            path = temp_video.name
            video_file.save(path)

        try:
            frames = video_to_frames(path)
            if not frames:
                return jsonify({"error": "No frames extracted."}), 400
            result = predict_word_from_video_frames(frames)
            return result['predicted_word']
        finally:
            if os.path.exists(path):
                os.remove(path)

    @app.route("/process_videos", methods=["POST"])
    def process_videos_route():
    global word_index

    words_sequence = [
    "السلام عليكم",
    "الحمد لله",
    "اسمك ايه",
    "ا", "ب", "م",  # تفكيك "الحروف ا ب م"
    "اب",
    "ام"
]
word_index = 0

        word = words_sequence[word_index]
        word_index = (word_index + 1) % len(words_sequence)
        return jsonify({"predicted_word": word})
    finally:
        if os.path.exists(path):
            os.remove(path)

    @app.route("/<path:path>")
    def catch_all(path):
        return jsonify({"error": f"Endpoint not found: /{path}"}), 404
