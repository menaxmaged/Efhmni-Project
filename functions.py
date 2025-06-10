import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- Model and Actions Configuration ---
# Load the pre-trained model for dynamic sign language recognition.
# This is done once when the server starts.
try:
    # MODIFICATION: Added compile=False to handle version mismatch errors.
    # This is safe for inference as compilation is only needed for training.
    model = load_model('DynamicModel.h5', compile=False)
    print("Dynamic gesture recognition model loaded successfully.")
except Exception as e:
    print(f"Error loading DynamicModel.h5: {e}")
    model = None

# Define the labels for the actions the model can recognize.
actions = np.array([
    "أسمك ايه ؟", "الحمد لله تمام", "بتشتغل إيه ؟", "بكام فلوس",
    "تيجي معايا ؟", "جيران", "صديق", "عائلة", "عريس", 
    "مشاء الله", "مع السلامة", "وحشتني"
])

# --- Mediapipe Initialization ---
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, holistic_model):
    """
    Processes an image with the Mediapipe Holistic model.
    
    Args:
        image (np.array): The input frame from the video.
        holistic_model: The Mediapipe Holistic instance.

    Returns:
        tuple: A tuple containing the processed image and the detection results.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """
    Extracts and flattens keypoints for pose, left hand, and right hand.

    Args:
        results: The detection results from Mediapipe.

    Returns:
        np.array: A flattened array of all keypoints.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

def video_to_frames(video_path, frame_interval=1):
    """
    Extracts frames from a video file at a specified interval.

    Args:
        video_path (str): The path to the video file.
        frame_interval (int): The interval at which to capture frames.

    Returns:
        list: A list of frames as NumPy arrays (in BGR format).
    """
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

def predict_from_frames(frames, threshold=0.7):
    """
    Processes video frames, predicts a sequence of actions, and forms a sentence.

    Args:
        frames (list): A list of video frames.
        threshold (float): The minimum confidence required to accept a prediction.

    Returns:
        dict: A dictionary containing the final predicted sentence.
    """
    if model is None:
        return {"predicted_sentence": "خطأ: النموذج لم يتم تحميله بنجاح."}
    
    sequence = []
    predictions = []
    sentence = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame in frames:
            # Perform detection
            _, results = mediapipe_detection(frame, holistic)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-60:]  # Keep the last 60 frames

            # Predict only when we have a full sequence
            if len(sequence) == 60:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = actions[np.argmax(res)]
                
                # Smooth predictions and build the sentence
                if res[np.argmax(res)] > threshold:
                    if not sentence or predicted_action != sentence[-1]:
                        sentence.append(predicted_action)

    # Return the final sentence
    final_sentence = " ".join(sentence) if sentence else "عذرا، لم يتم التعرف على الجملة."
    return {"predicted_sentence": final_sentence}
