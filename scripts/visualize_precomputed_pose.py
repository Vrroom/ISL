import cv2
import pickle
from PIL import Image
import random
import argparse
import mediapipe as mp
import numpy as np
from typing import List, Mapping, Optional, Tuple, Union
import isl_utils as islutils
from isl_utils import Wrapper

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Visualize poses')
    parser.add_argument('--pose_dir', type=str, required=True, help='Directory containing pose sequences')
    parser.add_argument('--metadata_file', type=str, required=True, help='File containing metadata')
    parser.add_argument('--pose_hash', default=None, type=str, help='(Optional) Hash of the pose sequence')
    parser.add_argument('--vis_bounds', action='store_true', default=False, help='(Optional) Visualize bounds of the pose sequence')

    args = parser.parse_args()

    if args.pose_hash is None :
        # if pose_hash is not provided, pick a random pose from the pose directory
        pose_pickle = random.choice(list(islutils.allfiles(args.pose_dir)))
        pose_hash = islutils.getBaseName(pose_pickle)
    else :
        pose_hash = args.pose_hash
        pose_pickle = osp.join(args.pose_dir, f'{pose_hash}.pkl')

    metadata = islutils.get_metadata_by_hash(args.metadata_file, pose_hash)
    width, height = metadata['width'], metadata['height']

    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    
    xs, ys = [], []

    for i in range (len(pose_sequence)) :
        landmark_list = Wrapper(dict(landmark=[Wrapper(_) for _ in pose_sequence[i]['landmarks']]))
        for idx, landmark in enumerate(landmark_list.landmark) :
            landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
            if landmark_px:
                xs.append(landmark_px[0])
                ys.append(landmark_px[1])

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    for i in range(len(pose_sequence)) :
        image = np.ones((height, width, 3))
        landmark_list = Wrapper(dict(landmark=[Wrapper(_) for _ in pose_sequence[i]['landmarks']]))
            
        if args.vis_bounds :
            # Draw bounds on the pose sequence
            cv2.line(image, (minx, miny), (maxx, miny), (0,255,0),1)
            cv2.line(image, (minx, miny), (minx, maxy), (0,255,0),1)
            cv2.line(image, (minx, maxy), (maxx, maxy), (0,255,0),1)
            cv2.line(image, (maxx, miny), (maxx, maxy), (0,255,0),1)

        mp_drawing.draw_landmarks(
                image,
                landmark_list,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow(f'Pose | {pose_hash}', islutils.aspectRatioPreservingResize(image, 512))

         # reak loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
 
