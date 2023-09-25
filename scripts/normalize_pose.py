import cv2
import pickle
from PIL import Image
import random
import argparse
import os
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import List, Mapping, Optional, Tuple, Union
import math
from copy import deepcopy
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

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Visualize random pose sequence')
    parser.add_argument('--pose_dir', type=str, help='Directory containing pose sequences')
    parser.add_argument('--metadata_file', type=str, help='File containing metadata')
    args = parser.parse_args()

    pose_pickle = random.choice(list(islutils.allfiles(args.pose_dir)))
    pose_hash = islutils.getBaseName(pose_pickle)
    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    metadata = islutils.get_metadata_by_hash(args.metadata_file, pose_hash)
    width, height = metadata['width'], metadata['height']
    n_pose_sequence = normalize_pose_sequence(pose_sequence, width, height)

    # visualize stuff
    fig, ax = plt.subplots()
    scatter = ax.scatter([1.5, -1.5], [1.5, -1.5])

    def init():
        ax.set_aspect('equal', 'box')
        return scatter,

    def update(frame):
        x_data = [n_pose_sequence[frame]['landmarks'][j]['x'] for j in range(len(n_pose_sequence[frame]['landmarks']))]
        y_data = [n_pose_sequence[frame]['landmarks'][j]['y'] for j in range(len(n_pose_sequence[frame]['landmarks']))]
        scatter.set_offsets(list(zip(x_data, y_data)))
        return scatter,

    print('Pose hash ...', pose_hash)
    print('Pose length ...', len(n_pose_sequence))
    ani = FuncAnimation(fig, update, frames=range(len(n_pose_sequence)), init_func=init, blit=True)
    plt.show()

