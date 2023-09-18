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

def video_downloader (data) :
    link, dir_path = data
    if any([_.startswith(link) for _ in os.listdir(dir_path)]) :
        return
    dataset_name = osp.split(dir_path)[1]
    with open(osp.join('video_jsons', dataset_name, f'{link}.json')) as fp : 
        duration = float(json.load(fp).get('duration', 1e10))
    if duration > 30 : 
        return
    try :
        cmd = ["yt-dlp", '-o', f'{dir_path}/{link}', f'https://www.youtube.com/watch?v={link}']
        subprocess.call(cmd)
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
    list(pmap(video_downloader, data, chunksize=500))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos.")
    parser.add_argument("--video_links_path", type=str, help="Path to the file containing video links")
    parser.add_argument("--save_path", type=str, help="Path where the processed data will be saved")
    args = parser.parse_args()
    main(args.video_links_path, args.save_path)


