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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def isclose(x, y, atol=1e-8, rtol=1e-5):
    """
    np.isclose replacement.

    Based on profiling evidence, it was found that the
    numpy equivalent is very slow. This is because np.isclose
    converts the numbers into internal representation and is
    general enough to work on vectors. We need this function
    to only work on numbers. Hence this faster alternative.
    """
    return abs(x - y) <= atol + rtol * abs(y)

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

class BBox :
    """
    Standard representation for Axis Aligned Bounding Boxes.

    Different modules have different but equivalent representations
    for bounding boxes. Some represent them as the top left corner
    along with height and width while others represent them as the
    top left corner and the bottom left corner. This class unifies
    both representations so that we can write bounding box methods
    in a single consistent way.
    """
    def __init__ (self, x, y, X, Y, w, h) :
        self.x = x
        self.y = y
        self.X = X
        self.Y = Y
        self.w = w
        self.h = h
        self.assertConsistent()

    def assertConsistent (self) :
        assert(self.X >= self.x)
        assert(self.Y >= self.y)
        assert(isclose(self.X - self.x, self.w))
        assert(isclose(self.Y - self.y, self.h))

    def iou (self, that) :
        if (self | that).isDegenerate() \
                or (self & that).isDegenerate() :
            return 0
        intersection = (self & that).area()
        union = (self | that).area()
        return intersection / (union + 1e-5)

    def isDegenerate (self) :
        return isclose(self.w, 0) and isclose(self.h, 0)

    def area (self) :
        if self.isDegenerate() :
            return 0
        else :
            return self.w * self.h

    def center (self) :
        return complex(
            (self.x + self.X) / 2,
            (self.y + self.Y) / 2
        )

    def __eq__ (self, that) :
        return isclose(self.x, that.x) \
                and isclose(self.X, that.X) \
                and isclose(self.y, that.y) \
                and isclose(self.Y, that.Y)

    def __mul__ (self, s) :
        """
        Scale bounding box by post multiplying by a constant

        A new box is made, scaled with respect to its origin.
        """
        return self.scaled(s, origin='center')

    def __or__ (self, that) :
        """ Union of two boxes """
        x = min(self.x, that.x)
        y = min(self.y, that.y)
        X = max(self.X, that.X)
        Y = max(self.Y, that.Y)
        return BBox(x, y, X, Y, X - x, Y - y)

    def __and__ (self, that) :
        """ Intersection of two boxes """
        x = max(self.x, that.x)
        y = max(self.y, that.y)
        X = min(self.X, that.X)
        Y = min(self.Y, that.Y)
        if y > Y: y = Y
        if x > X : x = X
        return BBox(x, y, X, Y, X - x, Y - y)

    def __contains__ (self, that) :
        return (self.x <= that.x <= that.X <= self.X \
                and self.y <= that.y <= that.Y <= self.Y) \
                and not self == that

    def __truediv__ (self, that):
        """ View of the box normalized to the coordinates of that box """
        nx = (self.x - that.x) / that.w
        ny = (self.y - that.y) / that.h
        nX = (self.X - that.x) / that.w
        nY = (self.Y - that.y) / that.h
        nw = nX - nx
        nh = nY - ny
        return BBox(nx, ny, nX, nY, nw, nh)

    def normalized (self) :
        """ Convert this box into the closest fitting square box """
        d = max(self.w, self.h)
        nx = self.x - (d - self.w) / 2
        ny = self.y - (d - self.h) / 2
        nX = nx + d
        nY = ny + d
        return BBox(nx, ny, nX, nY, d, d)

    def tolist (self, alternate=False) :
        if not alternate :
            return [self.x, self.y, self.w, self.h]
        else :
            return [self.x, self.y, self.X, self.Y]

    def __repr__ (self) :
        x = self.x
        y = self.y
        X = self.X
        Y = self.Y
        w = self.w
        h = self.h
        return f'BBox(x={x}, y={y}, X={X}, Y={Y}, w={w}, h={h})'

    def __xor__ (self, that) :
        """ check whether boxes are disjoint """
        b1 = sg.box(self.x, self.y, self.X, self.Y)
        b2 = sg.box(that.x, that.y, that.X, that.Y)
        return b1.disjoint(b2)

    def rotated (self, degree, pt=None) :
        if pt is None:
            pt = sg.Point(0, 0)
        else :
            pt = sg.Point(pt.real, pt.imag)
        x, y, X, Y = sa.rotate(self.toShapely(), degree, origin=pt).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def translated (self, tx, ty=0) :
        x, y, X, Y = sa.translate(self.toShapely(), tx, ty).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def scaled (self, sx, sy=None, origin=sg.Point(0, 0)) :
        if sy is None :
            sy = sx
        x, y, X, Y = sa.scale(self.toShapely(), sx, sy, origin=origin).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def skewX (self, xs) :
        x, y, X, Y = sa.skew(self.toShapely(), xs=xs).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def skewY (self, ys) :
        x, y, X, Y = sa.skew(self.toShapely(), ys=ys).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def toShapely (self) :
        return sg.Polygon([
            (self.x, self.y),
            (self.x, self.Y),
            (self.X, self.Y),
            (self.X, self.y)
        ])

def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height) :
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def get_metadata_by_hash(file_path, target_hash):
    df = pd.read_csv(file_path)
    row = df[df['hash'] == target_hash]
    return row.iloc[0].to_dict() if not row.empty else None

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

def getBaseName(fullName) :
    return osp.splitext(osp.split(fullName)[1])[0]

def load_random_pose (file_list): 
    # pick some random file
    pose_pickle = random.choice(file_list)
    pose_hash = getBaseName(pose_pickle)
    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    metadata = get_metadata_by_hash(args.metadata_file, pose_hash)
    return pose_sequence, metadata

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Create dataset for rest pose detection')
    parser.add_argument('--pose_dir', type=str, help='Directory containing pose sequences')
    parser.add_argument('--metadata_file', type=str, help='File containing metadata')
    parser.add_argument('--dataset_dir', type=str, help='Where to save the dataset')
    parser.add_argument('--class_size', type=int, help='Number of samples per class')
    parser.add_argument('--seq_len', type=int, help='Number of frames for pose subsequence')
    args = parser.parse_args()

    all_pose_files = list(allfiles(args.pose_dir))
    
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

    

