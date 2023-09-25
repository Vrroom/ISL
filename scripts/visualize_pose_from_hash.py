import cv2
import pickle
from PIL import Image
import random
import argparse
import os
import os.path as osp
import mediapipe as mp
import numpy as np
import pandas as pd
import isl_utils as islutils
from isl_utils import Wrapper

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Visualize pose sequence')
    parser.add_argument('--pose_hash', type=str, help='Hash of the pose sequence')
    parser.add_argument('--metadata_file', type=str, help='File containing metadata')
    args = parser.parse_args()

    pose_hash = args.pose_hash
    metadata = islutils.get_metadata_by_hash(args.metadata_file, pose_hash)
    width, height = metadata['width'], metadata['height']

    with open(f'../poses/{pose_hash}.pkl', 'rb') as fp : 
        pose_sequence = pickle.load(fp)

    for i in range(len(pose_sequence)) :
        landmark_list = Wrapper(dict(landmark=[Wrapper(_) for _ in pose_sequence[i]['landmarks']]))

        image = np.zeros((height, width, 3))
        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(
                image,
                landmark_list,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow(f'Pose | {pose_hash}', islutils.aspectRatioPreservingResize(image, 512))

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 

