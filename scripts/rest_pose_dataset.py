import cv2
from tqdm import tqdm
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
from copy import deepcopy
import shapely.geometry as sg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import isl_utils as islutils
from isl_utils import Wrapper, BBox


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height) :
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def normalize_pose_sequence (pose_sequence, width, height): 
    """
    Normalize the pose sequence so that the whole sequence 
    sits snuggly in a [-1, 1] by [-1, 1] box.
    """
    # calculate the fitting box
    xs, ys = [], []
    n_pose_sequence = deepcopy(pose_sequence)
    for i in range(len(pose_sequence)) :
        landmark_list = Wrapper(dict(landmark=[Wrapper(_) for _ in pose_sequence[i]['landmarks']]))
        for idx, landmark in enumerate(landmark_list.landmark) :
            landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
            n_pose_sequence[i]['landmarks'][idx]['x'] = landmark_px[0]
            n_pose_sequence[i]['landmarks'][idx]['y'] = landmark_px[1]
            xs.append(landmark_px[0])
            ys.append(landmark_px[1])
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    box = BBox(minx, miny, maxx, maxy, maxx - minx, maxy - miny)
    # now fit a square shaped box
    nbox = box.normalized()
    center = nbox.center()
    cx, cy = center.real, center.imag
    side = nbox.h
    # transform the pose sequence
    for i in range(len(n_pose_sequence)) :
        for j in range(len(n_pose_sequence[i]['landmarks'])) :
            n_pose_sequence[i]['landmarks'][j]['x'] -= cx
            n_pose_sequence[i]['landmarks'][j]['x'] *= 2.0 / side
            n_pose_sequence[i]['landmarks'][j]['y'] -= cy
            n_pose_sequence[i]['landmarks'][j]['y'] *= -2.0 / side
    return n_pose_sequence


def load_random_pose (file_list): 
    # pick some random file
    pose_pickle = random.choice(file_list)
    pose_hash = islutils.getBaseName(pose_pickle)
    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    metadata = islutils.get_metadata_by_hash(args.metadata_file, pose_hash)
    return pose_sequence, metadata

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Create dataset for rest pose detection')
    parser.add_argument('--pose_dir', type=str, help='Directory containing pose sequences')
    parser.add_argument('--metadata_file', type=str, help='File containing metadata')
    parser.add_argument('--dataset_dir', type=str, help='Where to save the dataset')
    parser.add_argument('--class_size', type=int, help='Number of samples per class')
    parser.add_argument('--seq_len', type=int, help='Number of frames for pose subsequence')
    args = parser.parse_args()

    all_pose_files = list(islutils.allfiles(args.pose_dir))
    
    # non-rest poses
    non_rest_poses = [] 
    for i in tqdm(range(args.class_size)) : 
        while True: 
            try:
                pose_sequence, metadata = load_random_pose(all_pose_files)
                width, height = metadata['width'], metadata['height']
                # sample a sequence, somewhere in the middle.
                st = random.randint(args.seq_len, len(pose_sequence) - 2 * args.seq_len)
                n_pose_sequence = normalize_pose_sequence(pose_sequence, width, height)[st:st+args.seq_len]
                # add pose sequence 
                non_rest_poses.append([])
                for frame in range(args.seq_len) : 
                    xy_data = [[pt['x'], pt['y']] for pt in n_pose_sequence[frame]['landmarks']] 
                    non_rest_poses[-1].append(xy_data) 
                break
            except Exception :
                pass
    # convert to numpy array
    non_rest_poses = np.array(non_rest_poses)

    # rest poses
    rest_poses = [] 
    for i in tqdm(range(args.class_size)) : 
        while True : 
            try :
                pose_sequence, metadata = load_random_pose(all_pose_files)
                width, height = metadata['width'], metadata['height']
                # sample a sequence, always from the start
                n_pose_sequence = normalize_pose_sequence(pose_sequence, width, height)[:args.seq_len]
                # add pose sequence 
                rest_poses.append([])
                for frame in range(args.seq_len) : 
                    xy_data = [[pt['x'], pt['y']] for pt in n_pose_sequence[frame]['landmarks']] 
                    rest_poses[-1].append(xy_data) 
                break
            except Exception : 
                pass

    # convert to numpy array
    rest_poses = np.array(rest_poses)

    print('Saving non-rest poses of shape', non_rest_poses.shape)
    np.save(osp.join(args.dataset_dir, 'non_rest_poses.npy'), non_rest_poses)

    print('Saving rest poses of shape', rest_poses.shape)
    np.save(osp.join(args.dataset_dir, 'rest_poses.npy'), rest_poses)

    

