import argparse
from tqdm import tqdm
import multiprocessing as mp
import os
import os.path as osp
import urllib.request
import zipfile
from functools import wraps
import hashlib

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

def skip_if_processed(task_file="/tmp/completed_tasks.txt"):
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


@skip_if_processed()
def calculate_md5(file_path):
    try : 
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            hash_md5.update(f.read())
        return hash_md5.hexdigest()
    except Exception :
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create md5sum table videos')
    parser.add_argument('--video_dir', required=True, help='Directory containing videos')
    parser.add_argument('--table_path', required=True, help='Path of the table')
    args = parser.parse_args()

    video_files = list(allfiles(args.video_dir))
    md5sums = list(pmap(calculate_md5, video_files))

    # remove the files which couldn't be computed
    video_files = [v for v, md5 in zip(video_files, md5sums) if md5 is not None]
    md5sums = [md5 for md5 in md5sums if md5 is not None]

    with open(args.table_path, 'w+') as fp : 
        fp.write('\n'.join([f'{a},{b}' for a, b in zip(video_files, md5sums)]))
