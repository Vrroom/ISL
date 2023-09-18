import argparse
import multiprocessing as mp
from tqdm import tqdm
import os
import os.path as osp
import subprocess 
import json

def pmap(function, items, chunksize=None) :
    """ parallel mapper using Pool with progress bar """
    cpu_count = mp.cpu_count()
    if chunksize is None :
        chunksize = len(items) // (cpu_count * 5)
    chunksize = max(1, chunksize)
    with mp.Pool(cpu_count) as p :
        mapper = p.imap(function, items, chunksize=chunksize)
        return list(tqdm(mapper, total=len(items)))

def json_downloader (data) :
    link, dir_path = data
    try :
        cmd = ["yt-dlp", "--skip-download", "-J", f'https://www.youtube.com/watch?v={link}']
        output = subprocess.run(cmd, capture_output=True, text=True).stdout
        data = json.loads(output)
        json_file_path = osp.join(dir_path, f"{link}.json")
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception :
        pass

def main(video_links_path, save_path):
    with open(video_links_path, 'r') as f:
        video_links = f.readlines()
        video_links = [_.strip() for _ in video_links]
        
    base_name = osp.splitext(osp.basename(video_links_path))[0]
    dir_path = osp.join(save_path, base_name)
    
    os.makedirs(dir_path, exist_ok=True)
    # Your logic here
    data = list(zip(video_links, [dir_path] * len(video_links)))
    list(pmap(json_downloader, data, chunksize=5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download JSON of YouTube video links.")
    parser.add_argument("--video_links_path", type=str, help="Path to the file containing video links")
    parser.add_argument("--save_path", type=str, help="Path where the processed data will be saved")
    args = parser.parse_args()
    main(args.video_links_path, args.save_path)

