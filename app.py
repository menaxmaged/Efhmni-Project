from flask import Flask, request, jsonify
from pyngrok import ngrok  # Correct Ngrok module
from functions import video_to_frames, predict_from_frames  # Import from video_processing

# Set up Flask app
app = Flask(__name__)

def start_ngrok(port: int):
    """
    Starts Ngrok tunnel for the specified port and returns the public URL.
    """
    print(" * Starting ngrok tunnel...")
    public_url = ngrok.connect(port)
    print('{}'.format(public_url))
    return public_url

# Set the configuration for Flask to avoid issues with non-ASCII characters
app.config['JSON_AS_ASCII'] = False

@app.route("/")
def home():
    return "<p>Egyptian Sign Language Translator</p>"

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    Handle the uploaded video, process it, and return the prediction result.
    """
    video_file = request.files['video']

    # Save the video file to disk
    video_path = 'video.mp4'
    video_file.save(video_path)

    # Extract frames from the video
    frames = video_to_frames(video_path)
    print("Number of frames: ", len(frames))

    # Process frames and get prediction
    result = predict_from_frames(frames)

    return result

@app.route('/upload', methods=['GET'])
def upload():
    """
    Provide the HTML form to upload a video for translation.
    """
    return '''
        <h1>مترجم لغة الإشارة المصرية</h1>
        <form method="POST" action="/process_video" enctype="multipart/form-data">
            <label>اختر فيديو (.mp4):</label><br><br>
            <input type="file" name="video" accept="video/mp4" required><br><br>
            <input type="submit" value="ابدأ الترجمة">
        </form>
    '''

@app.route('/<path:path>')
def catch_all(path):
    return f'No such page: {path}', 404

# Manual setup for Ngrok and Flask run
# Start Ngrok tunnel on port 5050
start_ngrok(5050)

if __name__ == "__main__":

    # Run Flask app on port 5000 (default port for `flask run`)
    app.run(port=5050)
