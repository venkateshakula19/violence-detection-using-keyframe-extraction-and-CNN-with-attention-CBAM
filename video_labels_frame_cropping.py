import os
import numpy as np
import cv2

def load_videos(video_dir):
    videos = []
    labels = []

    for filename in os.listdir(video_dir):
        if filename.endswith('.avi'):
            video_path = os.path.join(video_dir, filename)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            videos.append(frames)
            # Assign label based on filename or a separate label file
            labels.append(get_label_from_filename(filename))

    return np.array(videos, dtype=object), np.array(labels)

def get_label_from_filename(filename):
    # Example: Extract label from filename (e.g., 'class1_video1.mp4' or using a label file)
    if 'newfi' in filename:
        return 1
    else:
        return 0  # Handle unknown labels

# Example usage
video_dir = '/content/drive/MyDrive/output_keyframes_videos_data'
X_train, y_train = load_videos(video_dir)
print(len(X_train))
print(len(y_train))

# Example preprocessing (resizing frames)
def preprocess_videos(videos, target_height, target_width):
    preprocessed_videos = []

    for video in videos:
        preprocessed_frames = []
        for frame in video:
          if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
          resized_frame = cv2.resize(frame, (target_width, target_height))
          preprocessed_frames.append(resized_frame)
        preprocessed_videos.append(preprocessed_frames)

    return np.array(preprocessed_videos)

# Example usage
target_height, target_width = 224, 224  # Target frame size
X_train = preprocess_videos(X_train, target_height, target_width)
