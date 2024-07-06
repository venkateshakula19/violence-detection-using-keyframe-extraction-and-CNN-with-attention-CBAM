
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import zoom
import skvideo.io
import random
from matplotlib import pyplot as plt
import os
import tensorflow as tf

# Function to calculate MS-SSIM between two frames
def calculate_msssim(frame1, frame2, scales=5):
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    msssim = np.array([])
    mcs = np.array([])

    for _ in range(scales):
        ssim_map, cs_map = ssim(frame1, frame2, channel_axis=2, full=True)
        msssim = np.append(msssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())

        if _ < scales - 1:
            frame1 = zoom(frame1, (0.5, 0.5, 1), order=1)
            frame2 = zoom(frame2, (0.5, 0.5, 1), order=1)

    msssim_product = np.prod(mcs[:scales - 1]**weight[:scales - 1])
    return (msssim_product * (msssim[scales - 1]**weight[scales - 1]))


def read_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def extract_key_frames(video_path):
    #print(threshold)
    video = read_video(video_path)
    #video = cv2.VideoCapture(video_path)
    #skvideo.io.vread(video_path)

    key_frames = [video[0]]  # key_frames is a list of video frames and now having the first frame of the video
    prev_frame = video[0] # prev_frame holds the previous frame of a video
    count_key=0
    fr_no = 0
    scores_ssim = []
    #read frame by frame of input video and fill scores_ssim list with scores
    for frame in video[1:]:
      #printing the ssim score of two frames
      similarity = calculate_msssim(prev_frame, frame)
      print("frame number", fr_no)
      fr_no += 1
      #print("-")
      #print(similarity)

      #add that score to score_ssim list to compare with other scores
      scores_ssim.append(similarity)
      prev_frame = frame

    c_list = scores_ssim[:]
    result = find_17th_smallest(c_list)
    print("17th key frame ms-ssim score ")
    #print(result)
    #print("below are the ms-ssim scores of 16 key frames")

    print("total frames using fr_no", fr_no)

    #for index, value in enumerate(scores_ssim):
    #print(f"Index: {index}, Value: {value}")
    #print(result)
    #print("checking scores list")
    #print(scores_ssim)
    print("key frames selected in preprocessing")
    temp=0
    for h in scores_ssim:
      if temp < fr_no  and h < result:
        #temp += 1
        #append that particular frame to key_frames
        key_frames.append(video[temp + 1])
        #print(temp)
        #print(h)
        #temp += 1
        #print(h)
        count_key += 1
      temp += 1
    print(f"number of key frames {count_key}")
    return key_frames

def partition(lst, low, high):
    pivot_index = random.randint(low, high)
    pivot_value = lst[pivot_index]
    lst[pivot_index], lst[high] = lst[high], lst[pivot_index]
    i = low - 1

    for j in range(low, high):
        if lst[j] <= pivot_value:
            i += 1
            lst[i], lst[j] = lst[j], lst[i]

    lst[i + 1], lst[high] = lst[high], lst[i + 1]
    return i + 1

def quickselect(lst, low, high, k):
    if low <= high:
        pivot_index = partition(lst, low, high)

        if pivot_index == k:
            return lst[pivot_index]
        elif pivot_index < k:
            return quickselect(lst, pivot_index + 1, high, k)
        else:
            return quickselect(lst, low, pivot_index - 1, k)

def find_17th_smallest(lst):
    k = 16  # 17th smallest ssim score
    if 0 <= k < len(lst):
        result = quickselect(lst, 0, len(lst) - 1, k)
        return result
    else:
        # list with less than 16 elements
        return None

    return np.array(key_frames)


def save_key_frames(key_frames, output_path):
    skvideo.io.vwrite(output_path, key_frames)


def read_video_from_directory(directory_path):
    video_files = [file for file in os.listdir(directory_path) if file.endswith(".avi")]
    #i = 0
    videos = []
    #morph_videos_path = "/content/drive/MyDrive/morph_HF videos"
    for video_file in video_files:
        video_path = os.path.join(directory_path, video_file)
        #video = cv2.VideoCapture(video_path)
        #morph_videos_path = "/content/drive/MyDrive/morph_HF videos/video_file"
        key_frames = extract_key_frames(video_path)

        opening_frames = []
        #output_video_path = "/content/drive/MyDrive/morph_HF videos"
        binary_frames = []

        #vid = skvideo.io.vread(output_video_path)
        prev_frame = key_frames[0]
        threshold_value = 128
        for frame in key_frames[1:]:
          cv2_imshow(frame1)
          
        videos.append({
            'video_path': video_path,
            'frames': key_frames
        })

        # video.release()

    return videos

#main function
if __name__ == "__main__":
    dataset_path = "/content/drive/MyDrive/videos_data"

    videos = read_video_from_directory(dataset_path)

    for video in videos:
      opening_frames = video['frames']
      output_video_path = f"/content/drive/MyDrive/output_keyframes_videos_data/{os.path.basename(video['video_path'])}"
      save_key_frames(opening_frames, output_video_path)
