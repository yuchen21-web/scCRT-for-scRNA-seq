import argparse
import random
from scCRTUtils import *
from Estimate import Estimate_time

'''
dataset: PTDM
start_cluster = 1
dataset_path = 'data/PTDM.csv'
dataset_label_path = 'data/PTDM_label.csv'

dataset: binary_tree_8
start_cluster=0
dataset_path = 'data/binary_tree_8.csv'
dataset_label_path = 'data/binary_tree_8_label.csv'
model_path = 'data/binary_tree_8_model.pkl'
'''

def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("--dataset_path", type=str, default='data/PTDM.csv',
                       help="path of dataset")
    train.add_argument("--dataset_label_path", type=str, default='data/PTDM_label.csv',
                       help="path of the label")

    train.add_argument("--model_path", type=str, default=None,
                       help="the saved trained model path")
    train.add_argument("--device", type=str, default='cpu')
    train.add_argument("--hidden_size", type=int, default=128)
    train.add_argument("--output_size", type=int, default=16)
    train.add_argument("--k_adj", type=int, default=20,
                       help="the number of selected cell to construct positive set")
    train.add_argument("--k_traj", type=int, default=20,
                       help="the number of neighbour to construct trajectory connect matrix")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--start_cluster", type=int, default=1,
                       help="the start of cluster")
    train.add_argument("--tau", type=float, default=0,
                       help="the hyperparameter to controls softness")
    train.add_argument("--sample_type", type=str, default='random',
                       help="the sample type in training using positive set")

    return train

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':

    set_seed(1)
    args = setup_args().parse_args()
    ############################## 1. load data ##############################
    data_counts, cell_labels, cells, name2ids, ids2name, cell_times, pre_infos = getInputData(args.dataset_path,
                                                                                              args.dataset_label_path)
    ############################## 2. preprocess and train ##############################
    if cell_labels is None:
        features, cell_labels = pre_process(data_counts, WithoutLabel=True)
    else:
        features = pre_process(data_counts)

    device = torch.device(args.device)
    y_features = trainingModel(features, cell_labels, args)

    ############################## 3. get trajectory ##############################
    # given the start node
    if pre_infos is not None:
        start_node = name2ids[str(pandas2ri.rpy2py(pre_infos[2])[0])]
    else:
        start_node=args.start_cluster

    # get the clusters
    trajectory_data = get_trajectory(cell_labels,
                y_features, ids2name, cells, start_node=start_node, norm=args.tau, k=args.k_traj)

    network = trajectory_data[0].values.T[:2]
    for [s,t] in zip(network[0], network[1]):
        print(f'{s} --> {t}')

    '''
    actual topology of PTDM:
    Progenitor trophoblast_Gjb3 high(Placenta) --> Labyrinthine trophoblast(Placenta)
    Progenitor trophoblast_Gjb3 high(Placenta) --> Spongiotrophoblast_Phlda2 high(Placenta)
    Spongiotrophoblast_Phlda2 high(Placenta) --> Spongiotrophoblast_Hsd11b2 high(Placenta)
    Progenitor trophoblast_Gjb3 high(Placenta) --> Spiral artery trophoblast giant cells(Placenta)
    
    actual topology of binary_tree_8:
    M1 --> M3
    M3 --> M2
    M3 --> M7
    M7 --> M8
    M7 --> M5
    M5 --> M4
    M5 --> M6
    '''