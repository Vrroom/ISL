from nanoGPT.model import GPT, GPTConfig
import argparse
from rest_pose_dataset import *
import torch
import isl_utils as islutils
import pickle
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

@torch.no_grad()
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

def precision(model, x, y, threshold, p=None) :
    """ Precision is the fraction of true positives among all predicted positives """
    if p is None:
        p = get_probs(model, x)[:, 1]
    pred = p >= threshold
    return ((pred == 1) & (y == 1)).sum() / pred.sum()

def recall(model, x, y, threshold, p=None) : 
    """ Recall is the fraction of true positives among all actual positives """
    if p is None:
        p = get_probs(model, x)[:, 1]
    pred = p >= threshold
    return ((pred == 1) & (y == 1)).sum() / y.sum()

def precision_recall (model, x, y) : 
    thresholds = np.linspace(0, 1, 10)
    precisions, recalls = [], []
    probs = get_probs(model, x)[:, 1]
    for t in thresholds: 
        precisions.append(precision(model, x, y, t, p=probs))
        recalls.append(recall(model, x, y, t, p=probs))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.scatter(recalls, precisions, label='Precision v Recall')
    plt.legend()
    plt.show()

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description="Evaluate rest pose classification model on a pose sequence.")
    parser.add_argument('--model_checkpoint', required=True, type=str, help="Path to the model checkpoint.")
    parser.add_argument('--pose_dir', required=True, type=str, help="Path to the pose file.")
    parser.add_argument('--metadata_file', required=True, type=str, help="Metadata file path.")
    parser.add_argument('--rest_pose_data_dir', default='../rest_pose_dataset/', type=str, help="Directory containing dataset")
    args = parser.parse_args()

    # load model, you can download the trained parameters from (https://github.com/Vrroom/ISL/releases/download/rest-pose-ckpt/ckpt.pt)
    model = load_model(args.model_checkpoint)

    # Here, I'm demonstrating model use. I'm using the validation split (hard coded).
    rp  = torch.from_numpy(np.load(f'{args.rest_pose_data_dir}/rest_poses.npy')).float()[9000:]
    nrp = torch.from_numpy(np.load(f'{args.rest_pose_data_dir}/non_rest_poses.npy')).float()[9000:]
    x = torch.cat((rp, nrp), 0)
    y = torch.tensor([1] * 1000 + [0] * 1000)
    # Here we have N = 2000 sequences of L = 5 poses each. Each pose has P = 33 key points and each key point is defined by D = 2
    # dimensions. Thus, it is a 4 dimensional array. Before doing prediction, we'll "flatten" the last 2 dimensions.
    # For a 2D array - [[1, 2], [3, 4, 5], [6, 7]], this operation would convert it into a 1D array - [1, 2, 3, 4, 5, 6, 7]
    N, L, P, D = x.shape 
    x = x.reshape(N, L, P * D) # this does the above "flattening operation" for all x[i][j], 0 <= i < N, 0 <= j < L
    probs = get_probs(model, x) # check out the logic in the get_probs function. I have tried to add details

    # Plot precision v recall at various thresholds.
    precision_recall(model, x, y)
    
    all_pose_files = list(islutils.allfiles(args.pose_dir))
    print(f"Total files to process: {str(len(all_pose_files))}")

    pose_probs = {}
    for count, pose_file in enumerate(all_pose_files) :  
        pose_seq, pose_meta = islutils.load_pose(pose_file, args.metadata_file)

        width, height = pose_meta['width'], pose_meta['height']
        n_pose_sequence = islutils.normalize_pose_sequence(pose_seq, width, height)

        # build a list that can be passed to get_probs
        frames = []
        seqLength = len(n_pose_sequence)
        for index in range (seqLength-4) :
            # Here we are sliding a window of length 5 over the input pose sequence and extracting them for prediction
            frames.append([])
            for i in range(5) :
                xy_data = [[pt['x'], pt['y']] for pt in n_pose_sequence[i+index]['landmarks']]
                frames[-1].append(xy_data)

        x = torch.tensor(frames)
        N, L, P, D = x.shape 
        x = x.reshape(N, L, P * D) 

        probs = get_probs(model, x)
        pose_probs[pose_file] = probs # maybe problems here of deep copy. 

        t = np.array(list(range(seqLength - 4))) / pose_meta['framerate']
        rest_pose_prob = probs[:, 1].detach().numpy()

        plt.plot(t, rest_pose_prob, label='Rest pose probability v time')
        plt.legend()
        plt.show()
