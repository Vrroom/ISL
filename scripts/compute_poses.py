# precompute and store poses somewhere
import hashlib
import pickle
from tqdm import tqdm
import multiprocessing as mp_
import csv
from functools import wraps
import cv2
import argparse
import os
import os.path as osp
import mediapipe as mp
from isl_utils import pmap, skip_if_processed


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
output_dir = None

@skip_if_processed()
def compute_poses_for_video (data) :
    try:
        video_path, video_hash = data
        if osp.exists(osp.join(output_dir, f'{video_hash}.pkl')) :
            return
        cap = cv2.VideoCapture(video_path)
        landmarks = []
        with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                # now append all the information
                landmark_list = results.pose_landmarks.landmark
                landmarks.append(dict(
                    timestamp=timestamp,
                    landmarks=[
                        dict(
                            visibility=landmark_list[i].visibility,
                            x=landmark_list[i].x,
                            y=landmark_list[i].y,
                            z=landmark_list[i].z
                        )
                        for i in range(len(landmark_list))
                    ]
                ))
        cap.release()
        cv2.destroyAllWindows()
        # now dump landmarks
        with open(osp.join(output_dir, f'{video_hash}.pkl'), 'wb') as fp: 
            pickle.dump(landmarks, fp)
    except Exception :
        pass

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Precompute poses for all our videos')
    parser.add_argument('--pose_dir', type=str, help='Directory to store poses to')
    parser.add_argument('--video_hash_file', type=str, help='File containing video hashes')
    args = parser.parse_args()

    output_dir = args.pose_dir
    with open(args.video_hash_file) as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        data = list(reader)
    list(pmap(compute_poses_for_video, data))
