from nanoGPT.model import GPT, GPTConfig
import argparse
from rest_pose_dataset import *
import torch
import isl_utils as islutils
import pickle
import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
from matplotlib.animation import FuncAnimation


def load_model (model_path) : 
    """ 
    Helper method to load our classifier
    """
    print(f"Loading model from {model_path}")
    # resume training from a checkpoint.
    checkpoint = torch.load(model_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    model_args = {}
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    model.eval()
    return model


def get_probs (model, data) :
    logits, _ = model(data)  # N examples by 2 array

    # Example:
    # Let's assume logits = torch.Tensor([[1, 1], [-1, 1], [0, 0]])
    # After applying exp(), x will be:
    # [[e^1,  e^1],    <- logits for [rest pose, non rest pose]
    #  [e^-1, e^1],    <- logits for [rest pose, non rest pose]
    #  [e^0,  e^0]]    <- logits for [rest pose, non rest pose]
    #
    # x becomes:
    # [[2.7183, 2.7183],
    #  [0.3679, 2.7183],
    #  [1.0000, 1.0000]]
    #
    # After summing each row and dividing, probs becomes:
    # [[0.5,    0.5],    <- probabilities for [rest pose, non rest pose]
    #  [0.1192, 0.8808], <- probabilities for [rest pose, non rest pose]
    #  [0.5,    0.5]]    <- probabilities for [rest pose, non rest pose]
    #
    # Now each row in probs sums to 1 and all values are between 0 and 1

    # logits are raw numbers that the neural network outputs. These can be positive or negative.
    # We want to interpret these as probabilities. Probabilities, like percentages, have the property
    # that they are positive and for each example, they should sum to 1.
    #
    # A simple way to make numbers positive is to exponentiate them. e^x is always positive for all x
    x = torch.exp(logits)

    # Now to make sure that each row sums to 1, we'll divide by the total sum of each row
    probs = x / torch.sum(x, 1, keepdim=True)  # here 1 means sum across rows. 0 would mean sum across columns (which we don't want to do because ...
    # ... entries in a column come from different examples)
    return probs


def visualise_video(pose_probe, pose_xvals, video_dir) :
    pose_file, xvalues = next(iter(pose_xvals.items()))
    print (pose_file)
    probs = pose_probe[pose_file]

    #because pose_file is a pickel, and the video file has a different extension but the same base name
    # we'll need the below logic to build the complete file name and splitext method here is of not much use.
    video_file = islutils.getBaseName(pose_file).split('.')[0]
    video_file = video_dir + "/" + video_file + ".mov"
    if (os.path.exists(video_file) == False) :
        video_file = video_dir + "/" + video_file + ".mvk"
        if (os.path.exists(video_file) == False) :
            video_file = video_dir + "/" + video_file + ".mp4"
            if (os.path.exists(video_file) == False) :
                print("improper video file. Not able to visualise")
                return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Initialize the video display
    im = ax1.imshow(np.zeros((480, 640, 3), dtype=np.float32))
    ax2.scatter([0, max(xvalues)], [0, 1])
    ln, = ax2.plot([], [], 'r')

    cap = cv2.VideoCapture(video_file)
    xdata = []
    ydata = []
    prob_yvalues = probs[:,0].detach().numpy()

    def init () :
        return im, ln
    
    def update (frame) :
        ret, video_frame = cap.read()
        if ret:
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            im.set_array(video_frame)

        xdata.append(xvalues[frame])
        ydata.append(prob_yvalues[frame])
        ln.set_data(xdata, ydata)

        return im, ln

    ani = FuncAnimation(fig, update, list(range(len(xvalues))), init_func=init, blit=True, repeat=False)
    plt.show()

    cap.release()
    cv2.destroyAllWindows()

    # Wait for a key press and break if 'q' is pressed
    while(True) :
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description="Evaluate model on a pose_sequence.")
    parser.add_argument('--model_checkpoint', required=True, type=str, help="Path to the model checkpoint.")
    parser.add_argument('--pose_dir', required=True, type=str, help="Path to the pose file.")
    parser.add_argument('--metadata_file', required=True, type=str, help="Metadata file path.")
    parser.add_argument('--video_dir', required=True, type=str, help="Video directory file path.")
    args = parser.parse_args()

    # load model, you can download the trained parameters from (https://github.com/Vrroom/ISL/releases/download/rest-pose-ckpt/ckpt.pt)
    model = load_model(args.model_checkpoint)

    # Here, I'm demonstrating model use. I'm using the training dataset that I created using your normalization code.
    
    # pdb.set_trace()

    x = torch.from_numpy(np.load('../rest_pose_dataset/rest_poses.npy')).float()[:1000]
    # Here we have N = 1000 sequences of L = 5 poses each. Each pose has P = 33 key points and each key point is defined by D = 2
    # dimensions. Thus, it is a 4 dimensional array. Before doing prediction, we'll "flatten" the last 2 dimensions.
    # For a 2D array - [[1, 2], [3, 4, 5], [6, 7]], this operation would convert it into a 1D array - [1, 2, 3, 4, 5, 6, 7]
    N, L, P, D = x.shape 
    x = x.reshape(N, L, P * D) # this does the above "flattening operation" for all x[i][j], 0 <= i < N, 0 <= j < L
    probs = get_probs(model, x) # check out the logic in the get_probs function. I have tried to add details
    
    # load pose sequence pickle file (load_pose)
    all_pose_files = list(islutils.allfiles(args.pose_dir))
    print(f"total files to process: " + str(len(all_pose_files)))

    pose_probs = {}
    pose_xvalues = {}

    count = 0
    for pose_file in all_pose_files :  
        count = count + 1
        pose_seq, pose_meta = islutils.load_pose2(pose_file, args.metadata_file)

        width, height = pose_meta['width'], pose_meta['height']
        n_pose_sequence = islutils.normalize_pose_sequence(pose_seq, width, height)
        #build a list that can be passed to get_probs

        frames = []
            
        seqLength = len(n_pose_sequence)
        print(seqLength)
        for index in range (seqLength-4) :
            frames.append([])
            for i in range(5) :
                xy_data = [[pt['x'], pt['y']] for pt in n_pose_sequence[i+index]['landmarks']]
                frames[-1].append(xy_data)

        tense =  torch.tensor(frames)

        N, L, P, D = tense.shape 
        tense = tense.reshape(N, L, P * D) 

        probs = get_probs(model,tense)
        pose_probs[pose_file] = probs # maybe problems here of deep copy. 

        xvalues = list(range(seqLength-4))
        xvalues = [xvalue/30 for xvalue in xvalues]
        pose_xvalues[pose_file] = xvalues
        if (count == 10) :
            # breaking because the model crashes for larger count of the files. Needs to be fixed
            break

    visualise_video(pose_probs, pose_xvalues, args.video_dir)    

