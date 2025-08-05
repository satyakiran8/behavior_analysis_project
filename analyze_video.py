import cv2
import numpy as np
from fer import FER
import os

# =============================================================================
# INSTALLATION GUIDE
# =============================================================================
# Before running this script, you must install the required libraries.
# Make sure you are in your activated virtual environment.
# Run the following command in your terminal:
#
# pip install fer tensorflow moviepy
#
# You will also need to download two XML files from the OpenCV GitHub repository
# for face detection.
#
# 1. Download the following two XML files:
#    - Face Detector: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
#    - Eye Detector: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
#
# 2. Save both files in the same folder as this Python script.
#
# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
# The video path has been set to 'loughing2.mp4' directly.
# Please ensure this video file is in the same directory as this script.
VIDEO_PATH = "shouting.mp4"

# The script will now automatically determine the video duration and
# dynamically calculate FRAME_SKIP to ensure a consistent analysis time.
#
# This value will be calculated dynamically based on video length.
FRAME_SKIP = 1

# =============================================================================
# INITIALIZE EMOTION DETECTOR AND CASCADE CLASSIFIERS
# =============================================================================
# Initialize the FER (Facial Expression Recognition) emotion detector
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

# =============================================================================
# MAIN PROCESSING LOGIC
# =============================================================================
try:
    # Check if the video file exists at the specified path
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: The video file '{VIDEO_PATH}' was not found.")
        print(f"The script is currently running from: {os.getcwd()}")
        print("Please ensure the video file is in this directory or provide the full path to the file.")
        exit()
        
    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        exit()
    
    # Get video frame rate and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the video duration in seconds
    ANALYSIS_DURATION_SECONDS = int(total_frames / fps) if fps > 0 else 0
    
    if ANALYSIS_DURATION_SECONDS == 0:
        print("Error: Could not determine video duration. Please check the video file.")
        exit()

    # --- Dynamic FRAME_SKIP adjustment based on a formula ---
    # We aim to process a target number of frames for a representative sample,
    # ensuring the analysis time is consistent regardless of video length.
    TARGET_PROCESSED_FRAMES = 500
    
    if total_frames > TARGET_PROCESSED_FRAMES:
        FRAME_SKIP = int(total_frames / TARGET_PROCESSED_FRAMES)
    else:
        # For short videos, analyze every frame.
        FRAME_SKIP = 1
        
    # Ensure FRAME_SKIP is always at least 1 to avoid division by zero.
    FRAME_SKIP = max(1, FRAME_SKIP)
        
    frames_to_analyze = total_frames
    
    print(f"Analyzing the entire video, which is {ANALYSIS_DURATION_SECONDS} seconds long...")
    print("A summary will appear shortly.")

    # Define counters and dictionaries for behavioral analysis
    face_frames = 0
    total_frames_processed = 0
    emotion_counts = {
        "happy": 0, "sad": 0, "angry": 0, "fear": 0, "surprise": 0, "neutral": 0, "disgust": 0
    }
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= frames_to_analyze:
            break  # End of video file or reached the analysis duration

        frame_count += 1
        
        # Skip frames based on the dynamically set FRAME_SKIP value
        if frame_count % FRAME_SKIP != 0:
            continue
            
        total_frames_processed += 1

        # Convert frame to grayscale for cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            face_frames += 1
            
            for (x, y, w, h) in faces:
                # Use the FER detector to find emotions
                results = emotion_detector.detect_emotions(frame[y:y+h, x:x+w])
                
                if results:
                    # Get the dominant emotion and its score
                    dominant_emotion, score = emotion_detector.top_emotion(frame[y:y+h, x:x+w])
                    
                    # Increment the counter for the detected emotion
                    emotion_counts[dominant_emotion] += 1
    
    # =============================================================================
    # GENERATE AND DISPLAY FINAL BEHAVIORAL SUMMARY
    # =============================================================================
    # Calculate total emotions detected for normalization
    total_emotions_detected = sum(emotion_counts.values())

    if total_emotions_detected == 0:
        print("No emotions were detected in the video. Please check your video file and path.")
        exit()

    # Calculate psychological score and percentages for each emotion
    psychological_score = (emotion_counts['happy'] + emotion_counts['neutral']) / total_emotions_detected * 100
    
    # Calculate the percentage for each emotion, ensuring they sum to 100%
    emotion_percentages = {
        emotion: (count / total_emotions_detected) * 100
        for emotion, count in emotion_counts.items()
    }
    
    # Get frame dimensions for the summary screen.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, temp_frame = cap.read()
    if ret:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        frame_width = 640 # Default width

    # Prepare a simplified summary text with percentages
    summary_lines = [
        "--- Behavioral Analysis Summary ---",
        f"Video Duration: {ANALYSIS_DURATION_SECONDS} seconds",
        f"Total Frames Processed: {total_frames_processed}",
        f"Frames Skipped per Check: {FRAME_SKIP}",
        "",
        "--- Overall Psychological Score ---",
        f"Psychological Score: {psychological_score:.2f}%",
        "",
        "--- Emotion Breakdown (Percentages, sum to 100%) ---"
    ]
    for emotion, percentage in emotion_percentages.items():
        summary_lines.append(f"{emotion.capitalize()}: {percentage:.2f}%")

    summary_lines.extend([
        "",
        "Press any key to exit..."
    ])
    
    # Dynamically calculate the required height for the summary screen
    line_height = 30
    padding = 50
    summary_height = (len(summary_lines) * line_height) + padding
    summary_image = np.zeros((summary_height, frame_width, 3), dtype=np.uint8)
    
    # Draw summary text onto the black background
    y_offset = 50
    for line in summary_lines:
        cv2.putText(summary_image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_offset += 30
    
    # Print the summary to the terminal as well for clarity
    print("\n" + "\n".join(summary_lines))
    
    # Display the final summary screen
    cv2.imshow("Behavioral Analysis (Advanced)", summary_image)
    cv2.waitKey(0)  # Wait for any key press to close the window

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release the video capture object and destroy all windows
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()