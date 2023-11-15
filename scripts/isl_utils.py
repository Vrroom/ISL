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
import cv2
from functools import lru_cache, reduce, partial
from more_itertools import flatten
from skimage import transform

""" Definition of some file/directory paths that are used over and over again """
ROOT = "../"
POSE_DIR = osp.join(ROOT, "poses")
VIDEO_DIR = osp.join(ROOT, "videos")
VIDEO_METADATA = osp.join(ROOT, "metadata/video_metadata.csv")
VIDEO_HASH_METADATA = osp.join(ROOT, "metadata/video_hashes.csv")
VIDEO_JSONS = osp.join(ROOT, 'video_jsons')
VIDEO_ISLRTC_JSONS = osp.join(VIDEO_JSONS, 'islrtc')
VIDEO_RKM_JSONS = osp.join(VIDEO_JSONS, 'rkm')
HASH_TO_TEXT_MAPPING = osp.join(ROOT, "metadata/sign_language_pose_mappings.csv")
""" end """

def config_plot(ax):
    """ Function to remove axis tickers and box around a given axis """
    ax.set_frame_on(False)
    ax.axis('off')

def implies(a, b) :
    return not a or b

def allFiles (directory) :
    """ List all files that are not directories in this directory """
    return filter(lambda x : not osp.isdir(x), allfiles(directory))

@lru_cache
def cached_read_csv(file_name) : 
    return pd.read_csv(file_name)

@lru_cache
def cached_read_csv_as_dict(file_name, k, v) : 
    df = pd.read_csv(file_name)
    return dict(zip(df[k], df[v]))

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

def split_video(input_path, output_path1, output_path2, split_frame):
    """ 
    Splits an input video into two videos. 

    The first video has frames [0, split_frame) and the other has the rest
    """
    cap = cv2.VideoCapture(input_path)
    video_format = 'mp4v' if 'webm' not in output_path1 else 'VP80'
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    
    # video metadata ...
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out1 = cv2.VideoWriter(output_path1, fourcc, fps, (frame_width, frame_height))
    out2 = cv2.VideoWriter(output_path2, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        stream = out1 if frame_count < split_frame else out2
        stream.write(frame)
        frame_count += 1

    cap.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

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

def load_pose2 (pose_pickle, metadata_file): 
    pose_hash = getBaseName(pose_pickle)
    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    metadata = get_metadata_by_hash2(metadata_file, pose_hash)
    return pose_sequence, metadata

def load_pose (pose_pickle): 
    pose_hash = getBaseName(pose_pickle)
    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    metadata = get_metadata_by_hash(pose_hash)
    return pose_sequence, metadata

def get_metadata_by_hash2(file_path, target_hash):
    df = pd.read_csv(file_path)
    row = df[df['hash'] == target_hash]
    return row.iloc[0].to_dict() if not row.empty else None

def get_metadata_by_hash(target_hash):
    df = cached_read_csv(VIDEO_METADATA)
    row = df[df['hash'] == target_hash]
    return row.iloc[0].to_dict() if not row.empty else None

def get_video_path_by_hash (target_hash):
    hash_to_path = cached_read_csv_as_dict(VIDEO_HASH_METADATA, 'hash', 'path')
    return osp.join(ROOT, hash_to_path[target_hash])

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
        return list(tqdm.tqdm(mapper, total=len(items)))

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

def merge_dicts (dicts) :
    return reduce(lambda x, y : {**x, **y}, dicts)

def aggregateDict (listOfDicts, reducer, keys=None, defaultGet=None) :
    """
    Very handy function to combine a list of dicts
    into a dict with the reducer applied by key.
    """
    def reducerWithDefault (lst) :
        try :
            return reducer(lst)
        except Exception :
            return lst
    if not isinstance(listOfDicts, list) :
        listOfDicts = list(listOfDicts)
    if keys is None :
        keys = list(set(flatten(map(deepKeys, listOfDicts))))
    aggregator = lambda key : reducerWithDefault(
        list(map(
            partial(deepGet, deepKey=key, defaultGet=defaultGet),
            listOfDicts
        ))
    )
    return deepDict(zip(keys, map(aggregator, keys)))

def dictmap (f, d) :
    new = dict()
    for k, v in d.items() :
        new[k] = f(k, v)
    return new

def deepKeys (dictionary) :
    """ iterate over keys of dict of dicts """
    stack = [((), dictionary)]
    while len(stack) > 0 :
        prevKeys, dictionary = stack.pop()
        for k, v in dictionary.items() :
            if isinstance(v, dict) :
                stack.append(((*prevKeys, k), v))
            else :
                yield (*prevKeys, k)

def deepGet (dictionary, deepKey, defaultGet=None) :
    """ get key in a dict of dicts """
    v = dictionary.get(deepKey[0], defaultGet)
    if isinstance(v, dict) and len(deepKey) > 1:
        return deepGet(v, deepKey[1:])
    else :
        return v

def deepDict (pairs) :
    """
    Create a deep dict a.k.a a dict of dicts
    where a key may be tuple
    """
    d = dict()
    for k, v in pairs :
        d_ = d
        for k_ in k[:-1] :
            if k_ not in d_ :
                d_[k_] = dict()
            d_ = d_[k_]
        d_[k[-1]] = v
    return d

def getAll(thing, key) :
    """
    Traverse a dict or list of dicts
    in preorder and yield all the values
    for given key
    """
    if isinstance(thing, dict) :
        if key in thing :
            yield thing[key]
        for val in thing.values() :
            yield from getAll(val, key)
    elif isinstance(thing, list) :
        for val in thing :
            yield from getAll(val, key)

def f7(seq):
    """ Copied from somewhere on Stack Overflow """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def perspective_multiply(M, x):
    """
    Apply a perspective transformation to a set of 2D points.

    Args:
        M (numpy.ndarray): A 3x3 matrix representing the perspective transformation.
                          The matrix should be in the form of a numpy array with shape [3, 3].
        x (numpy.ndarray): An array of 2D points to be transformed.
                          Each point is represented as a row in this array, with the array having shape [N, 2],
                          where N is the number of points.

    Returns:
        numpy.ndarray: An array of transformed 2D points, with shape [N, 2].
                      The transformation is applied using the matrix M, and the points are then converted back to non-homogeneous coordinates.
    """
    N, _ = x.shape
    x_hom = np.concatenate((x, np.ones((N, 1))), axis=1)
    y_hom = ((M @ x_hom.T).T)
    y = y_hom[:, :-1] / y_hom[:, -1:]
    return y

def rand_perspective_transform_pose_sequence (normed_pos_seq, scale=0.4) : 
    """ Applies random perspective transform on the normalized pose sequence and renormalizes the result """
    normed_warped_pose_seq = deepcopy(normed_pos_seq)

    src = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    dst = src + 0.4 * np.random.randn(*src.shape)

    rand_perp = transform.ProjectiveTransform()
    rand_perp.estimate(src, dst)

    L = len(normed_pos_seq)

    xs = list(flatten([[_['x'] for _ in normed_pos_seq[i]['landmarks']] for i in range(L)]))
    ys = list(flatten([[_['y'] for _ in normed_pos_seq[i]['landmarks']] for i in range(L)]))

    xs = np.array(xs)
    ys = np.array(ys)

    pts = np.stack((xs, ys)).T

    new_pts = perspective_multiply(rand_perp.params, pts)

    minx, maxx, miny, maxy = new_pts[:, 0].min(), new_pts[:, 0].max(), new_pts[:, 1].min(), new_pts[:, 1].max()
    box = BBox(minx, miny, maxx, maxy, maxx - minx, maxy - miny)
    # now fit a square shaped box
    nbox = box.normalized()
    center = nbox.center()
    cx, cy = center.real, center.imag
    side = nbox.h

    new_pts[:, 0] -= cx
    new_pts[:, 1] -= cy
    new_pts *= 2.0 / side
    new_pts = new_pts.reshape(L, -1, 2)

    for i in range(L) : 
        for j in range(33): 
            normed_warped_pose_seq[i]['landmarks'][j]['x'] = new_pts[i, j, 0]
            normed_warped_pose_seq[i]['landmarks'][j]['y'] = new_pts[i, j, 1]
    
    return normed_warped_pose_seq

