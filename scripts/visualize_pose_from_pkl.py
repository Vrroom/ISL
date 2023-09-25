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
from typing import List, Mapping, Optional, Tuple, Union
import math
from isl_utils import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Visualize random pose sequence')
    parser.add_argument('--pose_dir', type=str, help='Directory containing pose sequences')
    parser.add_argument('--metadata_file', type=str, help='File containing metadata')
    args = parser.parse_args()

    pose_pickle = random.choice(list(allfiles(args.pose_dir)))
    pose_hash = getBaseName(pose_pickle)
    metadata = get_metadata_by_hash(args.metadata_file, pose_hash)
    width, height = metadata['width'], metadata['height']

    image = np.ones((height, width, 3))

    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    
    idx_to_coordinates = {}
    idx = 0
    minx = miny = -1
    maxx = maxy = -1
 
    for i in range(len(pose_sequence)) :
        landmark_list = Wrapper(dict(landmark=[Wrapper(_) for _ in pose_sequence[i]['landmarks']]))

        image = np.ones((height, width, 3))
        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(
                image,
                landmark_list,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow(f'Pose | {pose_hash}', aspectRatioPreservingResize(image, 512))

         # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
 
