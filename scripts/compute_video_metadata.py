import cv2
import csv
import argparse
from tqdm import tqdm
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
import pandas as pd
import os
import os.path as osp

def get_video_info(video_path):
    scene_cuts = len(detect(video_path, AdaptiveDetector()))
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return fps, duration, width, height, scene_cuts

def main(input_csv, output_csv):
    done_hashes = set()
    if osp.exists(output_csv):
        df = pd.read_csv(output_csv)
        done_hashes = set(df['hash'])
    print('Already seen items:', len(done_hashes))
    with open(input_csv, 'r') as infile, open(output_csv, 'a+', newline='') as outfile:
        csv_reader = csv.DictReader(infile)
        fieldnames = ['hash', 'framerate', 'duration', 'width', 'height', 'scene_cuts']
        csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        if len(done_hashes) == 0 :
            csv_writer.writeheader()
        for row in tqdm(csv_reader):
            try :
                video_path = row['path']
                video_hash = row['hash']
                if video_hash in done_hashes:
                    continue
                fps, duration, width, height, scene_cuts = get_video_info(video_path)
                csv_writer.writerow({
                    'hash': video_hash, 
                    'framerate': fps, 
                    'duration': duration, 
                    'width': width, 
                    'height': height, 
                    'scene_cuts': scene_cuts
                })
            except Exception :
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', help='CSV containing path and hash')
    parser.add_argument('--output_csv', help='CSV containing video metadata')
    args = parser.parse_args()
    main(args.input_csv, args.output_csv)

