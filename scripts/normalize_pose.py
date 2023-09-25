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
from copy import deepcopy
import shapely.geometry as sg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import isl_utils as islutils

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

