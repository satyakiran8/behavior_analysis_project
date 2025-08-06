import cv2
import numpy as np
from fer import FER
import mediapipe as mp
import os
import time
from collections import defaultdict
import threading
import queue
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle

# =============================================================================
# INSTALLATION AND SETUP GUIDE
# =============================================================================
# Before running this script, you must install all required libraries.
# Make sure you are in your activated virtual environment.
# Run the following commands in your terminal:
#
# pip install fer mediapipe tensorflow opencv-python sounddevice librosa scikit-learn scipy
#
# You will also need to download two XML files from the OpenCV GitHub repository
# for face detection.
#
# 1. Face Detector: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
# 2. Eye Detector: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
#
# Save both files in the same folder as this Python script.
#
# =============================================================================
# REAL-TIME ANALYSIS CONFIGURATION
# =============================================================================
# --- Video Configuration ---
WEBCAM_INDEX = 0

# --- Audio Configuration ---
AUDIO_CHUNK_DURATION = 2  # Capture and analyze audio every 2 seconds
FS = 44100  # Sample rate
TEMP_AUDIO_PATH = "temp_audio_chunk.wav"

# --- Emotion Labels ---
EMOTION_LABELS = ['happy', 'sad', 'angry', 'fear', 'disgust', 'neutral', 'surprise']
# For vocal tone analysis, we'll map pitch to these categories
TONE_LABELS = ['calm', 'neutral', 'aroused']

# =============================================================================
# INITIALIZE DETECTORS AND ANALYZERS
# =============================================================================
# Facial Emotion Recognition
emotion_detector = FER(mtcnn=False)

# Hand Gesture Recognition with MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load OpenCV cascade classifiers for face and eye detection.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Check if the classifiers were loaded correctly
if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load cascade classifier XML files.")
    print("Please ensure 'haarcascade_frontalface_default.xml' and 'haarcascade_eye.xml'")
    print("are in the same directory as this script.")
    exit()

# --- Vocal Tone Analysis Setup ---
# This section handles the audio emotion model. A dummy model is created if
# pre-trained files are not found, so the script can still run.
try:
    # Attempt to load a pre-trained model for vocal emotion if it exists
    # Note: For 'calm', 'aroused', etc., you'd need a custom model trained on
    # these specific tones, as they aren't standard emotions.
    with open('speech_tone_model.pkl', 'rb') as file:
        audio_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        audio_scaler = pickle.load(file)
    is_audio_model_loaded = True
    print("Pre-trained audio tone model loaded successfully.")
except FileNotFoundError:
    print("Warning: Pre-trained audio tone model files not found.")
    print("A dummy model will be created. Vocal tone analysis results will be random.")
    print("To get real results, you would need to train and save your own model.")
    # Create a simple dummy model for the sake of running the code
    audio_model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    audio_scaler = StandardScaler()
    # Dummy data to "fit" the model so it can be used for prediction
    dummy_features = np.random.rand(1, 193)  # Example feature vector size
    audio_model.fit(dummy_features, ['neutral'])
    audio_scaler.fit(dummy_features)
    is_audio_model_loaded = True


# =============================================================================
# AUDIO PROCESSING HELPER FUNCTIONS
# =============================================================================
def extract_features(file_path):
    """Extracts features from an audio file."""
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=FS)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        features = np.hstack([mfccs, chroma, mel, contrast])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# =============================================================================
# THREADING FOR AUDIO ANALYSIS
# =============================================================================
audio_queue = queue.Queue()
vocal_tone_counts = defaultdict(int)
is_running = threading.Event()
is_running.set()

def audio_capture_and_analysis_thread():
    """Captures audio and places it in a queue for analysis."""
    print("Audio capture started...")
    
    while is_running.is_set():
        try:
            print("Recording audio chunk...")
            audio_data = sd.rec(int(FS * AUDIO_CHUNK_DURATION), samplerate=FS, channels=1, blocking=True)
            write(TEMP_AUDIO_PATH, FS, audio_data)
            audio_queue.put(TEMP_AUDIO_PATH)
        except Exception as e:
            print(f"Error during audio recording: {e}")
            break

def audio_analysis_thread():
    """Analyzes audio chunks from the queue."""
    global vocal_tone_counts
    while is_running.is_set():
        try:
            file_path = audio_queue.get(timeout=1)
            
            features = extract_features(file_path)
            
            if features is not None:
                # Scale the features
                scaled_features = audio_scaler.transform(features.reshape(1, -1))
                # Predict the tone (or emotion from the dummy model)
                tone = audio_model.predict(scaled_features)[0]
                vocal_tone_counts[tone] += 1
            
            os.remove(file_path) # Cleanup
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error during audio analysis: {e}")

# =============================================================================
# MAIN PROCESSING LOGIC
# =============================================================================
face_emotion_counts = defaultdict(int)
total_face_frames = 0
hand_gesture_count = 0
total_video_frames = 0
face_detected_in_frame = False
eyes_detected_in_frame = False

try:
    # Open the video source (webcam)
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {WEBCAM_INDEX}")
        exit()

    print("Starting comprehensive real-time analysis. Press 'q' to quit.")
    
    # Start the audio threads
    if is_audio_model_loaded:
        capture_thread = threading.Thread(target=audio_capture_and_analysis_thread)
        analysis_thread = threading.Thread(target=audio_analysis_thread)
        capture_thread.daemon = True
        analysis_thread.daemon = True
        capture_thread.start()
        analysis_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1) # Flip for a mirror-like effect
        total_video_frames += 1
        
        # Reset frame-level flags
        face_detected_in_frame = False
        eyes_detected_in_frame = False

        # --- Face and Eye Detection ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            face_detected_in_frame = True
            for (x, y, w, h) in faces:
                # Draw rectangle on face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Detect facial emotions
                results = emotion_detector.detect_emotions(frame[y:y+h, x:x+w])
                if results:
                    dominant_emotion, score = emotion_detector.top_emotion(frame[y:y+h, x:x+w])
                    face_emotion_counts[dominant_emotion] += 1
                    cv2.putText(frame, f"Emotion: {dominant_emotion.capitalize()}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                # Detect eyes within the face region
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) > 0:
                    eyes_detected_in_frame = True
                    for (ex, ey, ew, eh) in eyes:
                        # Draw rectangle around eyes
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # --- Hand Gesture Detection (MediaPipe) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        
        if hand_results.multi_hand_landmarks:
            hand_gesture_count += 1
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # --- Behavioral and Psychological Signals Inference ---
        # Note: These are simple heuristics. A real system would use a
        # more sophisticated model trained for these specific signals.
        attentiveness = "Attentive" if eyes_detected_in_frame else "Inattentive"
        hand_use = "Active" if hand_results.multi_hand_landmarks else "Still"
        
        # Calculate scores and percentages for the summary overlay
        total_audio_chunks = sum(vocal_tone_counts.values())
        
        face_percentages = {emotion: (count / (total_video_frames + 1)) * 100 for emotion, count in face_emotion_counts.items()}
        tone_percentages = {tone: (count / (total_audio_chunks + 1)) * 100 for tone, count in vocal_tone_counts.items()}

        # Combine scores into a simple psychological metric
        positive_face_score = face_percentages.get('happy', 0) + face_percentages.get('neutral', 0)
        
        # Vocal tone contribution (arbitrary weighting for demonstration)
        positive_vocal_score = tone_percentages.get('calm', 0) + tone_percentages.get('neutral', 0)
        
        # Hand gesture contribution (frequent gestures might indicate nervousness)
        hand_score = 100 - (hand_gesture_count / (total_video_frames + 1)) * 100 # Invert for 'stillness'

        overall_psychological_score = (positive_face_score * 0.5) + (positive_vocal_score * 0.4) + (hand_score * 0.1)

        # --- Overlay text on the video frame ---
        y_offset = 20
        cv2.putText(frame, "--- Comprehensive Analysis ---", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Overall Score: {overall_psychological_score:.2f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_offset += 40
        
        cv2.putText(frame, "Facial Analysis:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        for emotion, percentage in face_percentages.items():
            cv2.putText(frame, f"  {emotion.capitalize()}: {percentage:.2f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 25
            
        y_offset += 15
        cv2.putText(frame, "Vocal Tone Analysis:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        for tone, percentage in tone_percentages.items():
            cv2.putText(frame, f"  {tone.capitalize()}: {percentage:.2f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
            y_offset += 25
        
        y_offset += 15
        cv2.putText(frame, "Behavioral Signals:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, f"  Attentiveness: {attentiveness}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, f"  Hand Movements: {hand_use}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Comprehensive Behavioral Analysis (Press q to quit)', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Cleanup
    print("\nShutting down...")
    is_running.clear()
    
    # Wait for threads to finish
    if is_audio_model_loaded:
        if 'capture_thread' in locals() and capture_thread.is_alive():
            capture_thread.join()
        if 'analysis_thread' in locals() and analysis_thread.is_alive():
            analysis_thread.join()
        
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    if os.path.exists(TEMP_AUDIO_PATH):
        os.remove(TEMP_AUDIO_PATH)
