import argparse
from tqdm import tqdm
import multiprocessing as mp
import os
import urllib.request
import zipfile
from functools import wraps
import hashlib

output_dir = None

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
def download_and_extract_zip(url):
    global output_dir
    filename = url.split('/')[-1]
    
    # Download ZIP file
    zip_path = os.path.join(output_dir, filename)
    urllib.request.urlretrieve(url, zip_path)
    
    # Unzip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Remove zip
    os.remove(zip_path)

def download_and_unzip(url_list_file):
    global output_dir
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Read the URLs from the file and strip them
    with open(url_list_file, 'r') as f:
        urls = [_.strip() for _ in f.readlines()]

    list(pmap(download_and_extract_zip, urls))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the INCLUDE dataset')
    parser.add_argument('--url_list_file', required=True, help='Path to the file containing URLs.')
    parser.add_argument('--output_dir', required=True, help='Directory to store the downloaded and unzipped files.')
    
    args = parser.parse_args()
    output_dir = args.output_dir
    download_and_unzip(args.url_list_file)

