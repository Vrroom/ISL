# precompute and store poses somewhere
import hashlib
import pickle
from tqdm import tqdm
import multiprocessing as mp_
import csv
from functools import wraps
import cv2
from PIL import Image
import random
import argparse
import os
import os.path as osp
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
output_dir = None

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
    cpu_count = mp_.cpu_count()
    if chunksize is None :
        chunksize = len(items) // (cpu_count * 5)
    chunksize = max(1, chunksize)
    with mp_.Pool(cpu_count) as p :
        mapper = p.imap(function, items, chunksize=chunksize)
        return list(tqdm(mapper, total=len(items)))


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

@skip_if_processed()
def compute_poses_for_video (data) :
    try:
        video_path, video_hash = data
        if osp.exists(osp.join(output_dir, f'{video_hash}.pkl')) :
            return
        cap = cv2.VideoCapture(video_path)
        landmarks = []
        with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                # now append all the information
                landmark_list = results.pose_landmarks.landmark
                landmarks.append(dict(
                    timestamp=timestamp,
                    landmarks=[
                        dict(
                            visibility=landmark_list[i].visibility,
                            x=landmark_list[i].x,
                            y=landmark_list[i].y,
                            z=landmark_list[i].z
                        )
                        for i in range(len(landmark_list))
                    ]
                ))
        cap.release()
        cv2.destroyAllWindows()
        # now dump landmarks
        with open(osp.join(output_dir, f'{video_hash}.pkl'), 'wb') as fp: 
            pickle.dump(landmarks, fp)
    except Exception :
        pass

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Precompute poses for all our videos')
    parser.add_argument('--pose_dir', type=str, help='Directory to store poses to')
    parser.add_argument('--video_hash_file', type=str, help='File containing video hashes')
    args = parser.parse_args()

    output_dir = args.pose_dir
    with open(args.video_hash_file) as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        data = list(reader)
    list(pmap(compute_poses_for_video, data))
