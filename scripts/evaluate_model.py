from nanoGPT.model import GPT, GPTConfig
import argparse
from rest_pose_dataset import *
import torch
import isl_utils as islutils
import pickle
# import numpy as np
import pdb
import matplotlib.pyplot as plt



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

def load_pose (pose_pickle): 
    # pick some random file
    pose_hash = islutils.getBaseName(pose_pickle)
    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    metadata = islutils.get_metadata_by_hash(args.metadata_file, pose_hash)
    return pose_sequence, metadata

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


if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description="Evaluate model on a pose_sequence.")
    parser.add_argument('--model_checkpoint', required=True, type=str, help="Path to the model checkpoint.")
    parser.add_argument('--pose_dir', required=True, type=str, help="Path to the pose file.")
    parser.add_argument('--metadata_file', required=True, type=str, help="Metadata file path.")
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
    count = 0
    for pose_file in all_pose_files :  
        count = count + 1
        pose_seq, pose_meta = islutils.load_pose(pose_file, args.metadata_file)

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
        print (tense.shape)

        N, L, P, D = tense.shape 
        tense = tense.reshape(N, L, P * D) 

        probs = get_probs(model,tense)
        pose_probs[pose_file] = probs # maybe problems here of deep copy. 

        xvalues = list(range(seqLength-4))
        xvalues = [xvalue/30 for xvalue in xvalues]
        if (count == 10) :
            break

    islutils.writeCSV("probs.csv", pose_probs)

#    plt.plot(xvalues, probs[:,0].detach().numpy())
#    plt.show()

    #pdb.set_trace()



    # normalize pose sequence (normalize_pose_sequence imported from rest_pose_dataset)

    # Let N be sequence length
    # start from i = 2, i = N - 3
    # pick a set of 5 frames centered at i. Two on left and two on right.
    # get the probability of rest pose for these 5 frames.
    # append it to a list
    # plot probability as a function of i.
