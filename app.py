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
    ngrok.set_auth_token("2N1uyElcHqbXEsvE6616QFzSn4W_6rZ1Ek8vBJNGsKXyRhZ3P")

    public_url = ngrok.connect(port, domain="tender-sculpin-badly.ngrok-free.app")
    #print('{}'.format(public_url))
    print(f"Ngrok tunnel with static domain is running at: {public_url}")

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
    try:
        # Step 1: Retrieve the uploaded video file
        video_file = request.files['video']
        print(f'\033[1m\033[32mReceived video file: {video_file.filename}\033[0m')
        print(f'\033[1m\033[34mFile type: {video_file.content_type}\033[0m')
        
        # Check if the file is empty
        if not video_file:
            raise ValueError("No video file uploaded")
        
        # Reading file size
        video_file.seek(0, 2)  # Seek to the end to get the size
        file_size = video_file.tell()
        print(f'\033[1m\033[34mFile size: {file_size} bytes\033[0m')
        video_file.seek(0)  # Reset file pointer to the beginning

        # Step 2: Save the video file to disk
        video_path = 'video.mp4'
        video_file.save(video_path)
        print(f'\033[1m\033[33mVideo saved at: {video_path}\033[0m')

        # Step 3: Extract frames from the video
        try:
            frames = video_to_frames(video_path)
            print(f'\033[1m\033[36mNumber of frames extracted: {len(frames)}\033[0m')
        except Exception as e:
            print(f'\033[1m\033[31mError extracting frames: {e}\033[0m')
            return jsonify({"error": f"Error extracting frames: {str(e)}"}), 500

        # Step 4: Process frames and get prediction
        try:
            print(f'\033[1m\033[35mProcessing frames...\033[0m')
            result = predict_from_frames(frames)
        except Exception as e:
            print(f'\033[1m\033[31mError processing frames: {e}\033[0m')
            return jsonify({"error": f"Error processing frames: {str(e)}"}), 500
        
        # Step 5: Print result of prediction
        print(f'\033[1m\033[32mPrediction result: {result}\033[0m')

        # Step 6: Return the prediction result
        return  result#jsonify({"result": result})

    except ValueError as e:
        print(f'\033[1m\033[31mError: {e}\033[0m')
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f'\033[1m\033[31mUnexpected error: {e}\033[0m')
        return jsonify({"error": "An unexpected error occurred."}), 500

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
