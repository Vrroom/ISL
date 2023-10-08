import argparse
import pickle
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import os.path as osp
import isl_utils as islutils
import torch  
from evaluate_rest_pose_model import load_model, get_probs  
import pandas as pd

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', required=True, help='Model checkpoint file')
    parser.add_argument('--seq_len', default=10, type=int, help='Sequence length')
    args = parser.parse_args()

    # get all hashes
    hashes = list(pd.read_csv(islutils.VIDEO_METADATA)['hash'])

    # Load model
    model = load_model(args.model_checkpoint)
    video_paths = [islutils.get_video_path_by_hash(_) for _ in hashes]
    rkm_pairs = [(a, b) for (a, b) in zip(hashes, video_paths) if 'rkm' in b]

    for hash_, video_path in tqdm(rkm_pairs) :
        try :
            # Load pose
            pose_file = osp.join(islutils.POSE_DIR, f'{hash_}.pkl')
            pose_seq, pose_meta = islutils.load_pose(pose_file)
            
            width, height = pose_meta['width'], pose_meta['height']
            n_pose_sequence = islutils.normalize_pose_sequence(pose_seq, width, height)

            # Build a list that can be passed to get_probs
            frames = []
            seqLength = len(n_pose_sequence)
            for index in range(seqLength - args.seq_len + 1):
                frames.append([])
                for i in range(args.seq_len):
                    xy_data = [[pt['x'], pt['y']] for pt in n_pose_sequence[i+index]['landmarks']]
                    frames[-1].append(xy_data)

            x = torch.tensor(frames)
            N, L, P, D = x.shape
            x = x.reshape(N, L, P * D)

            probs = get_probs(model, x)
            rest_pose_prob = probs[:, 1].detach().cpu().numpy() 
            # Apply a smoothing operation
            rest_pose_prob_sm = gaussian_filter1d(rest_pose_prob, 1)

            zipped_probs = list(zip(rest_pose_prob_sm, range(args.seq_len // 2, args.seq_len // 2 + N)))

            middle_third_st = seqLength // 3
            middle_third_en = 2 * seqLength // 3

            # Extract the probabilities for the middle 1/3rd
            middle_third_probs = [(prob, idx) for prob, idx in zipped_probs if middle_third_st <= idx <= middle_third_en]

            # Take the max probability
            max_prob, max_id = max(middle_third_probs)
            if max_prob > 0.1 : 
                print(f'Max ID = {max_id}, Half Pt = {seqLength // 2}, Hash = {hash_}')
                ext = osp.splitext(video_path)[1]
                path_a = osp.join(islutils.RKM_SPLIT_DIR, f'{hash_}_1{ext}')
                path_b = osp.join(islutils.RKM_SPLIT_DIR, f'{hash_}_2{ext}')
                islutils.split_video(video_path, path_a, path_b, max_id)
                with open(osp.join(islutils.POSES_RKM_SPLIT_DIR, f'{hash_}_1.pkl'), 'wb') as fp : 
                    pickle.dump(pose_seq[:max_id], fp) 
                with open(osp.join(islutils.POSES_RKM_SPLIT_DIR, f'{hash_}_2.pkl'), 'wb') as fp : 
                    pickle.dump(pose_seq[max_id:], fp) 
        except Exception :
            pass
