# scCRT-for-scRNA-seq


scCRT is a dimensionality reduction model for scRNA-seq trajectory inference.

Overall architecture of the scCRT model pipeline:

![Model](https://github.com/yuchen21-web/scCRT-for-scRNA-seq/blob/main/src/Model.png)


scCRT employs two feature learning components, a cell-level pairwise module and a cluster-level contrastive module, to learn accurate positional representations of cells conducive to inferring cell lineages. The cell-level module focuses on learning accurate cell representations in a reduced-dimensionality space while maintaining the cell–cell positional relationships in the original space. The cluster-level contrastive module uses prior cell state information to aggregate similar cells, preventing excessive dispersion in the low-dimensional space.


# Requirements
- python (for use)
  - python = 3.7
  - sklearn
  - numpy
  - pandas
  - torch
  - scanpy
  - scipy
  - [pcurvepy](https://github.com/mossjacob/pcurvepy) (for estimating pseudotime)
  - rpy2 (for evaluation using dyneval package)

- R (for evaluation)
  - [dynwrap](https://github.com/dynverse/dynwrap)
  - magrittr

# Tutorial

This is a running guide for scCRT using public datasets in our experments. Moreover, we provided a saved trained model and binary_tree_8 synthetic dataset to verify the effectiveness of the paper.

## 1. Follow the procedure below to perform scCRT on binary_tree_8 synthetic dataset with the jupyter or on [tutorial_scCRT.ipynb](https://github.com/yuchen21-web/scCRT-for-scRNA-seq/blob/main/src/tutorial_scCRT.ipynb)

### 1.1 Read the dataset.

```python
from scCRT.utils_func import *
from scCRT.Estimate import Estimate_time
device=torch.device('cpu')
```

```python
# real data and labels

# dataset_path = 'data/binary_tree_8.csv'
# dataset_label_path = 'data/binary_tree_8_label.csv'

# get_data
data_counts, cell_labels, cells, name2ids, ids2name, cell_times, pre_infos = getInputData(dataset_path, 
                                                                                          dataset_label_path)
```

### 1.2 preprocess the data.

```python
# If there is no cell labels of prior information, Louvain can be used for partitioning like PAGA
if cell_labels is None:
    features, cell_labels = pre_process(data_counts, WithoutLabel=True)
else:
    features = pre_process(data_counts)
```

### 1.3 perform scCRT to learn cell features.
```python
'''
Parameters
----------
input: 
features: the preprocessed expression matrix, [n_cell, n_genes], numpy.array
cell_labels: the label of cells, numpy.array

output: 
y_features: the learned cell features, [n_cell, feature_size]
'''

args.model_path = 'data/binary_tree_8_model.pkl'
y_features = trainingModel(features, cell_labels, args)
```

### 1.4 Infer trajectory.
```python
# given the start node
start_node=0

# get the clusters
trajectory_data = get_trajectory(cell_labels, 
            y_features, ids2name, cells, start_node=start_node, norm=False, k=10)
network = trajectory_data[0].values.T[:2]

for [s,t] in zip(network[0], network[1]):
    print(f'{s} --> {t}')
```
M1 --> M3  
M3 --> M2  
M3 --> M7  
M7 --> M5  
M5 --> M4  
M5 --> M6  
M7 --> M8  


## 2. Run a simple program directly with run_test.py on PTDM dataset

### 2.1 The environment

usee the provided requirement.txt or use conda as follow:

```shell
~$ conda create -n env_name python==3.9.0
~$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
~$ conda install scikit-learn==1.3.0
~$ pip install scanpy==1.9.9
~$ pip install pcurvepy2

scCRT$ python run_test.py
```
Progenitor trophoblast_Gjb3 high(Placenta) --> Labyrinthine trophoblast(Placenta)
Progenitor trophoblast_Gjb3 high(Placenta) --> Spongiotrophoblast_Phlda2 high(Placenta)
Spongiotrophoblast_Phlda2 high(Placenta) --> Spongiotrophoblast_Hsd11b2 high(Placenta)
Progenitor trophoblast_Gjb3 high(Placenta) --> Spiral artery trophoblast giant cells(Placenta)







