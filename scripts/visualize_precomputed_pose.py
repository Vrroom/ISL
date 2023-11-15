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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import os.path as osp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Visualize poses')
    parser.add_argument('--pose_dir', type=str, default=islutils.POSE_DIR, help='Directory containing pose sequences')
    parser.add_argument('--pose_hash', default=None, type=str, help='(Optional) Hash of the pose sequence')
    parser.add_argument('--vis_bounds', action='store_true', default=False, help='(Optional) Visualize bounds of the pose sequence')
    parser.add_argument('--normalize', action='store_true', default=False, help='(Optional) Normalize the pose in a [-1, 1] by [-1, 1] box. Uses matplotlib for visualizing. Incompatible with some other options')
    parser.add_argument('--perspective_augment', action='store_true', default=False, help='(Optional) Visualize perspective augmentation')

    args = parser.parse_args()

    if args.pose_hash is None :
        # if pose_hash is not provided, pick a random pose from the pose directory
        pose_pickle = random.choice(list(islutils.allfiles(args.pose_dir)))
        pose_hash = islutils.getBaseName(pose_pickle)
    else :
        pose_hash = args.pose_hash
        pose_pickle = osp.join(args.pose_dir, f'{pose_hash}.pkl')

    metadata = islutils.get_metadata_by_hash(pose_hash)
    width, height = metadata['width'], metadata['height']

    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    
    if args.normalize: 
        n_pose_sequence = islutils.normalize_pose_sequence(pose_sequence, width, height)

        if args.perspective_augment : 
            augs_to_show = 4
            
            augmented_pose_seqs = [islutils.rand_perspective_transform_pose_sequence(n_pose_sequence) for _ in range(augs_to_show)]
            fig, ax = plt.subplots(1, 1 + augs_to_show)

            [islutils.config_plot(_) for _ in ax] 

            scatters = [_.scatter([1.5, -1.5], [1.5, -1.5]) for _ in ax]

            def init() : 
                for _ in ax : 
                    _.set_aspect('equal', 'box')
                return scatters

            def update(frame):
                x_data = [n_pose_sequence[frame]['landmarks'][j]['x'] for j in range(len(n_pose_sequence[frame]['landmarks']))]
                y_data = [n_pose_sequence[frame]['landmarks'][j]['y'] for j in range(len(n_pose_sequence[frame]['landmarks']))]
                scatters[0].set_offsets(list(zip(x_data, y_data)))

                for i in range(augs_to_show) : 
                    x_data = [augmented_pose_seqs[i][frame]['landmarks'][j]['x'] for j in range(len(augmented_pose_seqs[i][frame]['landmarks']))]
                    y_data = [augmented_pose_seqs[i][frame]['landmarks'][j]['y'] for j in range(len(augmented_pose_seqs[i][frame]['landmarks']))]
                    scatters[i + 1].set_offsets(list(zip(x_data, y_data)))

                return scatters

            print('Pose hash ...', pose_hash)
            print('Pose length ...', len(n_pose_sequence))
            ani = FuncAnimation(fig, update, frames=range(len(n_pose_sequence)), init_func=init, blit=True, repeat=False)
            plt.show()

        else: 
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
            ani = FuncAnimation(fig, update, frames=range(len(n_pose_sequence)), init_func=init, blit=True, repeat=False)
            plt.show()

    else:
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
     
