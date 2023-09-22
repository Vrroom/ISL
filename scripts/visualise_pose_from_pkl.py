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


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def get_metadata_by_hash(file_path, target_hash):
    df = pd.read_csv(file_path)
    row = df[df['hash'] == target_hash]
    return row.iloc[0].to_dict() if not row.empty else None

class Wrapper :
    """ Just a wrapper class so that we can call the `drawing_landmark` function """
    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def __getattr__ (self, x) : 
        return self.dictionary.get(x)

    def HasField (self, x) :
        return x in self.dictionary

def listdir (path) :
    """
    Convenience function to get
    full path details while calling os.listdir

    Also ensures that the order is always the same.

    Parameters
    ----------
    path : str
        Path to be listed.
    """
    paths = [osp.join(path, f) for f in os.listdir(path)]
    paths.sort()
    return paths

def allfiles (directory) :
    """ List full paths of all files/directory in directory """
    for f in listdir(directory) :
        yield f
        if osp.isdir(f) :
            yield from allfiles(f)

def imgArrayToPIL (arr) :
    """ utility to convert img array to PIL """
    if arr.dtype in [np.float32, np.float64, float] :
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype in [np.int32, np.int64, int]:
        arr = arr.astype(np.uint8)
    assert(arr.dtype == np.uint8)
    chanType = "RGBA" if arr.shape[2] == 4 else "RGB"
    return Image.fromarray(arr, chanType)

def aspectRatioPreservingResize (arr, smaller_dim) :
    """ utility for resizing image """
    pil_img = imgArrayToPIL(arr)
    h, w = pil_img.size
    if h < w :
        h, w = smaller_dim, smaller_dim * w / h
    else :
        h, w = smaller_dim * h / w, smaller_dim
    h, w = int(h), int(w)
    resized = pil_img.resize((h, w))
    np_arr = np.array(resized).astype(arr.dtype)
    if arr.dtype in [float, np.float32, np.float64] :
        np_arr /= 255.0
    return np_arr


def getBaseName(fullName) :
    return osp.splitext(osp.split(fullName)[1])[0]

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
 
        landmark_list = Wrapper(dict(landmark=[Wrapper(_) for _ in pose_sequence[0]['landmarks']]))

        for idx, landmark in enumerate(landmark_list.landmark) :
            
            image_rows, image_cols, _ = image.shape
            landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                            image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
                if (minx == -1 or landmark_px[0] < minx) :
                    minx = landmark_px[0]
                elif (landmark_px[0] > maxx) :
                    maxx = landmark_px[0]
                if (miny == -1 or landmark_px[1] < miny) :
                    miny = landmark_px[1]
                elif (landmark_px[1] > maxy) :
                    maxy = landmark_px[1]

        cv2.line(image, (minx, miny), (maxx, miny), (0, 255, 0), 1)
        cv2.line(image, (minx, miny), (minx, maxy), (0,255,0),1)
        cv2.line(image, (minx, maxy), (maxx, maxy), (0,255,0),1)
        cv2.line(image, (maxx, miny), (maxx, maxy), (0,255,0),1)
    
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
 
