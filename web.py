import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# =============================================================================
# 1. MODEL AND CONFIGURATION LOADING
# =============================================================================

# --- Load the pre-trained model for static sign language recognition ---
try:
    # Load the model. `compile=False` is used for faster loading when only doing inference.
    model = load_model('arabic_sign_language_model.h5', compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("🚨 Please ensure 'arabic_sign_language_model.h5' is in the same directory.")
    model = None

# --- Define the class labels that correspond to the model's output ---
class_labels = [
    'ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain',
    'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra',
    'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw',
    'ya', 'yaa', 'zay'
]

# =============================================================================
# 2. IMAGE PREPROCESSING FUNCTION
# =============================================================================

def preprocess_frame(frame):
    """
    Preprocesses an image frame for the sign language model.
    The input frame is expected to be a NumPy array from OpenCV.
    """
    # Resize the image to the model's expected input size (224x224)
    img = cv2.resize(frame, (224, 224))
    # Convert the image to a NumPy array
    img_array = img_to_array(img)
    # Expand dimensions to create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the pixel values to be between 0 and 1
    img_array /= 255.0
    return img_array

# =============================================================================
# 3. REAL-TIME WEBCAM RECOGNITION
# =============================================================================

def start_webcam_recognition():
    """
    Starts the webcam, captures frames, and performs real-time prediction.
    """
    # Check if the model was loaded successfully before starting
    if model is None:
        print("🔴 Cannot start recognition because the model is not loaded.")
        return

    # Initialize the webcam. 0 is usually the default built-in webcam.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ Webcam started. Press 'q' to quit.")

    # Main loop to continuously get frames from the webcam
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for a more intuitive, mirror-like view
        frame = cv2.flip(frame, 1)

        # Define a Region of Interest (ROI) for the user to place their hand
        # This is a rectangle on the screen. We will only predict on this area.
        top, right, bottom, left = 100, 350, 400, 650
        cv2.rectangle(frame, (right, top), (left, bottom), (0, 255, 0), 2)
        
        # Extract the ROI from the frame
        roi = frame[top:bottom, right:left]

        # --- Prediction Logic ---
        if roi.size > 0:
            # Preprocess the ROI for the model
            preprocessed_roi = preprocess_frame(roi)
            
            # Perform the prediction
            prediction = model.predict(preprocessed_roi)
            
            # Get the predicted label and confidence score
            predicted_class_idx = np.argmax(prediction[0])
            predicted_label = class_labels[predicted_class_idx]
            confidence = np.max(prediction[0]) * 100  # As a percentage

            # --- Display the result on the screen ---
            # Create the text to display
            display_text = f"{predicted_label} ({confidence:.2f}%)"
            
            # Set up text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_pos = (right, top - 10) # Position text just above the ROI
            font_scale = 1
            font_color = (0, 255, 0) # Green
            line_type = 2

            # Draw the text on the main frame
            cv2.putText(frame, display_text, text_pos, font, font_scale, font_color, line_type)

        # Display the final frame with the ROI and prediction
        cv2.imshow("Sign Language Recognition - Press 'q' to quit", frame)

        # Check for user input to quit the application
        # `cv2.waitKey(1)` waits 1ms for a key press. `& 0xFF` is a standard bitmask.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("🔴 Webcam stopped and windows closed.")

# --- Main Execution ---
if __name__ == "__main__":
    start_webcam_recognition()
