import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

from lanedetect_helpers import process_image
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def lane_detect_images():
    test_data_dir = "test_images/"
    # Read Test images
    test_images = os.listdir(test_data_dir)
    for test_image in test_images:
        image = mpimg.imread(os.path.join(test_data_dir, test_image))
        final_image = process_image(image)

def lane_detect_videos():
    test_data_dir = "test_videos/"
    video_out_dir = "test_videos_output/"
    test_videos = os.listdir(test_data_dir)
    for test_video in test_videos:
        test_video_input = os.path.join(test_data_dir, test_video)
        test_video_output = os.path.join(video_out_dir, test_video)
        video_clip = VideoFileClip(test_video_input)
        video_frame = video_clip.fl_image(process_image)
        video_frame.write_videofile(test_video_output, audio=False)

if __name__ =="__main__":
    lane_detect_images()
    #lane_detect_videos()

