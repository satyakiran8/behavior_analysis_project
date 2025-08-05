import cv2
import numpy as np
from fer import FER
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
# Before running this script, you must install the required libraries.
# Make sure you are in your activated virtual environment.
# Run the following commands in your terminal:
#
# pip install fer tensorflow opencv-python sounddevice librosa scikit-learn scipy
#
# You will also need to download two XML files from the OpenCV GitHub repository
# for face detection.
#
# 1. Face Detector: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
# 2. Eye Detector: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
#
# Save both files in the same folder as this Python script.
#
# THIS SCRIPT DOES NOT USE DLIB AND DOES NOT REQUIRE `shape_predictor_68_face_landmarks.dat`.
# The face detection is now handled exclusively by OpenCV's Haar Cascades.
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

# =============================================================================
# INITIALIZE EMOTION DETECTOR AND AUDIO ANALYZER
# =============================================================================
# Initialize the FER (Facial Expression Recognition) emotion detector
# We pass `mtcnn=False` to ensure that FER uses a dlib-free face detector.
emotion_detector = FER(mtcnn=False)

# Create a placeholder model for audio emotion analysis.
# A real application would use a pre-trained model file.
# This dummy model ensures the script runs without errors.
# You can replace this with a model you have trained yourself.
try:
    # Attempt to load a pre-trained model if it exists
    with open('speech_emotion_model.pkl', 'rb') as file:
        audio_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        audio_scaler = pickle.load(file)
    is_audio_model_loaded = True
except FileNotFoundError:
    print("Warning: Pre-trained audio emotion model files not found.")
    print("A dummy model will be created. Audio analysis results will be random.")
    print("To get real results, you would need to train and save your own model.")
    # Create a simple dummy model for the sake of running the code
    audio_model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    audio_scaler = StandardScaler()
    # Dummy data to "fit" the model so it can be used for prediction
    dummy_features = np.random.rand(1, 193)  # Example feature vector size
    audio_model.fit(dummy_features, ['neutral'])
    audio_scaler.fit(dummy_features)
    is_audio_model_loaded = True

# Load the pre-trained cascade classifiers for face and eye detection.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Check if the classifiers were loaded correctly
if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load cascade classifier XML files.")
    print("Please ensure 'haarcascade_frontalface_default.xml' and 'haarcascade_eye.xml'")
    print("are in the same directory as this script.")
    exit()

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
audio_emotion_counts = defaultdict(int)
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
    global audio_emotion_counts
    while is_running.is_set():
        try:
            file_path = audio_queue.get(timeout=1)
            
            features = extract_features(file_path)
            
            if features is not None:
                # Scale the features
                scaled_features = audio_scaler.transform(features.reshape(1, -1))
                # Predict the emotion
                emotion = audio_model.predict(scaled_features)[0]
                audio_emotion_counts[emotion] += 1
            
            os.remove(file_path) # Cleanup
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error during audio analysis: {e}")

# =============================================================================
# MAIN PROCESSING LOGIC
# =============================================================================
video_emotion_counts = defaultdict(int)
total_video_frames = 0

try:
    # Open the video file
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {WEBCAM_INDEX}")
        exit()

    print("Starting real-time analysis. Press 'q' to quit.")
    
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

        # --- Video (Facial) Emotion Analysis ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                results = emotion_detector.detect_emotions(frame[y:y+h, x:x+w])
                
                if results:
                    dominant_emotion, score = emotion_detector.top_emotion(frame[y:y+h, x:x+w])
                    video_emotion_counts[dominant_emotion] += 1
                    total_video_frames += 1
        
        # --- Real-time display and summary calculation ---
        # Calculate video emotion percentages
        video_emotion_percentages = {}
        if total_video_frames > 0:
            total_video_emotions_detected = sum(video_emotion_counts.values())
            if total_video_emotions_detected > 0:
                video_emotion_percentages = {
                    emotion: (count / total_video_emotions_detected) * 100
                    for emotion, count in video_emotion_counts.items()
                }

        # Calculate audio emotion percentages
        audio_emotion_percentages = {}
        total_audio_emotions_detected = sum(audio_emotion_counts.values())
        if total_audio_emotions_detected > 0:
            audio_emotion_percentages = {
                emotion: (count / total_audio_emotions_detected) * 100
                for emotion, count in audio_emotion_counts.items()
            }
        
        # --- Calculate Overall Score ---
        video_happy_neutral = video_emotion_counts.get('happy', 0) + video_emotion_counts.get('neutral', 0)
        audio_happy_neutral = audio_emotion_counts.get('happy', 0) + audio_emotion_counts.get('neutral', 0)
        
        total_happy_neutral = video_happy_neutral + audio_happy_neutral
        total_combined_emotions = total_video_frames + total_audio_emotions_detected
        
        if total_combined_emotions > 0:
            overall_psychological_score = (total_happy_neutral / total_combined_emotions) * 100
        else:
            overall_psychological_score = 0
            
        # --- Overlay text on the video frame ---
        y_offset = 20
        cv2.putText(frame, "--- Real-time Analysis ---", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Overall Score: {overall_psychological_score:.2f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_offset += 40
        
        cv2.putText(frame, "Facial Emotions:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        for emotion, percentage in video_emotion_percentages.items():
            cv2.putText(frame, f"  {emotion.capitalize()}: {percentage:.2f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 25
            
        y_offset += 15
        cv2.putText(frame, "Vocal Emotions:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        for emotion, percentage in audio_emotion_percentages.items():
            cv2.putText(frame, f"  {emotion.capitalize()}: {percentage:.2f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
            y_offset += 25

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Analyzer (Press q to quit)', frame)

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
    
    if os.path.exists(TEMP_AUDIO_PATH):
        os.remove(TEMP_AUDIO_PATH)
