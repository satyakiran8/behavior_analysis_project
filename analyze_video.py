import cv2
import numpy as np
from fer import FER
import os
import time
from collections import defaultdict
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from pydub import AudioSegment
import pickle
import glob

# =============================================================================
# INSTALLATION AND SETUP GUIDE
# =============================================================================
# Before running this script, you must install the required libraries.
# Make sure you are in your activated virtual environment.
# Run the following commands in your terminal:
#
# pip install fer opencv-python pydub librosa scikit-learn
#
# You will also need to download two XML files from the OpenCV GitHub repository
# for face detection.
#
# 1. Face Detector: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
# 2. Eye Detector: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
#
# Save both files in the same folder as this Python script, along with your video file.
#
# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
VIDEO_PATH = "loughing2.mp4"
TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_CHUNKS_DIR = "audio_chunks"
AUDIO_CHUNK_DURATION_MS = 2000  # 2 seconds
FS = 44100  # Sample rate for audio processing

# =============================================================================
# INITIALIZE EMOTION DETECTORS
# =============================================================================
# Facial Expression Recognition (FER) detector
emotion_detector = FER(mtcnn=True)

# Load the pre-trained cascade classifiers for face and eye detection.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Check if the classifiers were loaded correctly
if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load cascade classifier XML files.")
    print("Please ensure 'haarcascade_frontalface_default.xml' and 'haarcascade_eye.xml'")
    print("are in the same directory as this script.")
    exit()

# Audio Emotion Analyzer
# This is a dummy model. A real application would require a pre-trained model.
try:
    with open('audio_emotion_model.pkl', 'rb') as file:
        audio_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        audio_scaler = pickle.load(file)
    is_audio_model_loaded = True
except FileNotFoundError:
    print("Warning: Pre-trained audio emotion model files not found.")
    print("A dummy model will be created. Audio analysis results will be random.")
    print("To get real results, you need to train and save your own model and scaler.")
    
    audio_model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    audio_scaler = StandardScaler()
    
    # Fit the dummy model and scaler with some random data so it can be used
    dummy_features = np.random.rand(1, 193)  # Example feature vector size
    audio_model.fit(dummy_features, ['neutral'])
    audio_scaler.fit(dummy_features)
    is_audio_model_loaded = True

# =============================================================================
# AUDIO PROCESSING HELPER FUNCTIONS
# =============================================================================
def extract_audio_from_video(video_path, output_path):
    """Extracts the audio track from a video file using pydub."""
    print("Extracting audio from video...")
    try:
        video = AudioSegment.from_file(video_path)
        audio = video.set_channels(1).set_frame_rate(FS).set_sample_width(2)
        audio.export(output_path, format="wav")
        print("Audio extraction complete.")
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

def extract_features(file_path):
    """Extracts a set of audio features using librosa."""
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=FS)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        features = np.hstack([mfccs, chroma, mel, contrast])
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# =============================================================================
# MAIN PROCESSING LOGIC
# =============================================================================
# Cleanup previous temporary files
if os.path.exists(TEMP_AUDIO_PATH):
    os.remove(TEMP_AUDIO_PATH)
if os.path.exists(AUDIO_CHUNKS_DIR):
    for file in glob.glob(os.path.join(AUDIO_CHUNKS_DIR, "*.wav")):
        os.remove(file)
    os.rmdir(AUDIO_CHUNKS_DIR)

# Extract audio from video and process it
if extract_audio_from_video(VIDEO_PATH, TEMP_AUDIO_PATH):
    # Split audio into chunks for analysis
    os.makedirs(AUDIO_CHUNKS_DIR, exist_ok=True)
    audio_file = AudioSegment.from_wav(TEMP_AUDIO_PATH)
    audio_length_ms = len(audio_file)
    
    audio_emotion_counts = defaultdict(int)
    print("Analyzing audio chunks...")
    for i, start_ms in enumerate(range(0, audio_length_ms, AUDIO_CHUNK_DURATION_MS)):
        end_ms = start_ms + AUDIO_CHUNK_DURATION_MS
        chunk = audio_file[start_ms:end_ms]
        chunk_path = os.path.join(AUDIO_CHUNKS_DIR, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")
        
        features = extract_features(chunk_path)
        if features is not None and is_audio_model_loaded:
            scaled_features = audio_scaler.transform(features.reshape(1, -1))
            emotion = audio_model.predict(scaled_features)[0]
            audio_emotion_counts[emotion] += 1
        
    print("Audio analysis complete.")
    os.remove(TEMP_AUDIO_PATH)
    for file in glob.glob(os.path.join(AUDIO_CHUNKS_DIR, "*.wav")):
        os.remove(file)
    os.rmdir(AUDIO_CHUNKS_DIR)
else:
    audio_emotion_counts = defaultdict(int)

# Process video frames
video_emotion_counts = defaultdict(int)
total_video_frames = 0
last_analysis_time = time.time()
analysis_interval = 0.5

try:
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        exit()

    print(f"\nStarting facial analysis of '{VIDEO_PATH}'.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_time = time.time()
        
        if current_frame_time - last_analysis_time >= analysis_interval:
            last_analysis_time = current_frame_time
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    results = emotion_detector.detect_emotions(frame[y:y+h, x:x+w])
                    if results:
                        dominant_emotion, score = emotion_detector.top_emotion(frame[y:y+h, x:x+w])
                        video_emotion_counts[dominant_emotion] += 1
                        total_video_frames += 1
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, f"{dominant_emotion.capitalize()}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Hybrid Emotion Analysis (Press q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # =============================================================================
    # GENERATE AND DISPLAY FINAL BEHAVIORAL SUMMARY
    # =============================================================================
    video_emotion_percentages = {}
    if total_video_frames > 0:
        total_video_emotions_detected = sum(video_emotion_counts.values())
        if total_video_emotions_detected > 0:
            video_emotion_percentages = {
                emotion: (count / total_video_emotions_detected) * 100
                for emotion, count in video_emotion_counts.items()
            }
            
    audio_emotion_percentages = {}
    total_audio_emotions_detected = sum(audio_emotion_counts.values())
    if total_audio_emotions_detected > 0:
        audio_emotion_percentages = {
            emotion: (count / total_audio_emotions_detected) * 100
            for emotion, count in audio_emotion_counts.items()
        }

    video_happy_neutral = video_emotion_counts.get('happy', 0) + video_emotion_counts.get('neutral', 0)
    audio_happy_neutral = audio_emotion_counts.get('happy', 0) + audio_emotion_counts.get('neutral', 0)
    
    total_happy_neutral = video_happy_neutral + audio_happy_neutral
    total_combined_emotions = total_video_frames + total_audio_emotions_detected
    
    if total_combined_emotions > 0:
        overall_psychological_score = (total_happy_neutral / total_combined_emotions) * 100
    else:
        overall_psychological_score = 0
            
    # Print the final summary to the console
    print("\n--- Hybrid Analysis Summary ---")
    print(f"Video: {VIDEO_PATH}")
    print(f"Total Video Frames Analyzed: {total_video_frames}")
    print(f"Total Audio Chunks Analyzed: {total_audio_emotions_detected}")
    print("\nFacial Emotions Detected:")
    for emotion, percentage in video_emotion_percentages.items():
        print(f"  - {emotion.capitalize()}: {percentage:.2f}%")
    print("\nVocal Emotions Detected:")
    for emotion, percentage in audio_emotion_percentages.items():
        print(f"  - {emotion.capitalize()}: {percentage:.2f}%")
    print(f"\nOverall Psychological Score (Happy/Neutral): {overall_psychological_score:.2f}%")
    
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
