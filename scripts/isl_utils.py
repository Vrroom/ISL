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

def get_metadata_by_hash(file_path, target_hash):
    df = pd.read_csv(file_path)
    row = df[df['hash'] == target_hash]
    return row.iloc[0].to_dict() if not row.empty else None

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
