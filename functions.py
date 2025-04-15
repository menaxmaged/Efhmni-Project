import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Load the model globally
model = load_model('DynamicModel.h5')

# Action labels
actions = np.array(["أسمك ايه ؟","الحمد لله تمام","بتشتغل إيه ؟","بكام فلوس",
                    "تيجي معايا ؟","جيران", "صديق", "عائلة", "عريس", 
                    "مشاء الله", "مع السلامة", "وحشتني"])

def video_to_frames(video_path, frame_interval=1):
    """
    Convert video to frames at a specified interval.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % frame_interval == 0:
            frames.append(frame)

    cap.release()
    return frames

def mediapipe_detection(image, model):
    """
    Perform Mediapipe detection on the image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """
    Extract keypoints from Mediapipe results.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def predict_from_frames(frames, threshold=0.7):
    """
    Process frames and predict actions using the loaded model.
    """
    sequence = []
    predictions = []
    sentence = []
    
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame in frames:
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-60:]

            if len(sequence) == 60:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

    return sentence[-1] if len(sentence) != 0 else "عذرا, لم يتم التحقق جيدا من الجملة."
