import os
import numpy as np
import os.path as osp
import json
import argparse
import isl_utils as islutils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete long videos')
    parser.add_argument('--video_dir', type=str, help='Directory containing videos')
    parser.add_argument('--json_dir', type=str, help='Directory containing jsons')
    parser.add_argument('--video_links_file', type=str, help='File containing video links')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    args = parser.parse_args()
    
    with open(args.video_links_file) as fp:
        links = [_.strip() for _ in fp.readlines()]

    videos_paths = islutils.listdir(osp.join(args.video_dir, args.dataset_name))
 
    for link in links : 
        json_path = osp.join(args.json_dir, args.dataset_name, f'{link}.json')
        with open(json_path) as fp:
            duration = float(json.load(fp).get('duration', 1e10))
        if duration > 30 :
            paths_to_remove = [_ for _ in videos_paths if link in _]
            [os.remove(_) for _ in paths_to_remove]

