""" This file contains utility functions that are used in the isl project """
import tqdm
import os
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image
import hashlib
import multiprocessing as mp
import shapely.geometry as sg
import shapely.affinity as sa
from functools import wraps
from copy import deepcopy
import math
import random
import pickle
import csv

""" Definition of some file/directory paths that are used over and over again """
ROOT = "../"
POSE_DIR = osp.join(ROOT, "poses")
VIDEO_DIR = osp.join(ROOT, "videos")
VIDEO_METADATA = osp.join(ROOT, "metadata/video_metadata.csv")
VIDEO_HASH_METADATA = osp.join(ROOT, "metadata/video_hashes.csv")
""" end """

def writeCSV(fname, dict) :
    """ Write dictionary as csv to a file """
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in dict.items():
            if isinstance(value, list) : 
                # spread the list
                writer.writerow([key, *value])
            else:
                writer.writerow([key, value])

def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height) :
    """
    Poses returned by mediapipe are normalized with respect to the image width and height
    so that they are in the range [0, 1] for both x and y coordinates. This does the inverse operation
    """
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def normalize_pose_sequence (pose_sequence, width, height): 
    """
    Normalize the pose sequence so that the whole sequence sits snuggly in a [-1, 1] by [-1, 1] box.
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

def getBaseName(fullName) :
    return osp.splitext(osp.split(fullName)[1])[0]

def load_random_pose (file_list): 
    # pick some random file
    pose_pickle = random.choice(file_list)
    pose_hash = getBaseName(pose_pickle)
    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    metadata = get_metadata_by_hash(pose_hash)
    return pose_sequence, metadata

def load_pose (pose_pickle): 
    pose_hash = getBaseName(pose_pickle)
    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    metadata = get_metadata_by_hash(pose_hash)
    return pose_sequence, metadata

def get_metadata_by_hash(target_hash):
    df = pd.read_csv(VIDEO_METADATA)
    row = df[df['hash'] == target_hash]
    return row.iloc[0].to_dict() if not row.empty else None

def get_video_path_by_hash (target_hash):
    df = pd.read_csv(VIDEO_HASH_METADATA)
    hash_to_path = dict(zip(df['hash'], df['path']))
    return hash_to_path[target_hash]

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

def skip_if_processed(task_file="completed_tasks.txt"):
    """
    Decorator to skip function execution if it has been previously processed.

    This is useful in data processing if program crashes and you want to restart
    without doing everything all over again.
    """

    # Initialize a set to hold completed task hashes
    completed_task_hashes = set()

    # Load completed task hashes from file
    if os.path.exists(task_file):
        with open(task_file, "r") as f:
            completed_task_hashes = set(line.strip() for line in f.readlines())

    def compute_hash(args):
        """Compute SHA-256 hash for the given arguments."""
        args_str = str(args)
        return hashlib.sha256(args_str.encode()).hexdigest()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_hash = compute_hash(args)

            # If the task is not yet completed, run it and mark as completed
            if task_hash not in completed_task_hashes:
                result = func(*args, **kwargs)

                # Update completed task hashes and save to file
                completed_task_hashes.add(task_hash)
                with open(task_file, "a") as f:
                    f.write(f"{task_hash}\n")

                return result
            else:
                print(f"Task with hash {task_hash} has already been processed, skipping.")
        return wrapper
    return decorator

def pmap(function, items, chunksize=None) :
    """ parallel mapper using Pool with progress bar """
    cpu_count = mp.cpu_count()
    if chunksize is None :
        chunksize = len(items) // (cpu_count * 5)
    chunksize = max(1, chunksize)
    with mp.Pool(cpu_count) as p :
        mapper = p.imap(function, items, chunksize=chunksize)
        return list(tqdm(mapper, total=len(items)))

class Wrapper :
    """ 
    Just a wrapper class for a dictionary so that we can 
    refer to its keys using the dot notation
    """

    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def __getattr__ (self, x) : 
        return self.dictionary.get(x)

    def HasField (self, x) :
        return x in self.dictionary

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
