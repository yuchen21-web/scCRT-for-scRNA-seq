# scCRT-for-scRNA-seq


scCRT is a dimensionality reduction model for scRNA-seq trajectory inference.

Overall architecture of the scCRT model pipeline:

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
  - dynbenchmark

# Tutorial

This is a running guide for scCRT using public datasets in our experments. Moreover, we provided a saved trained model and binary_tree_8 synthetic dataset to verify the effectiveness of the paper.

## 1. Follow the procedure below to perform scCRT on binary_tree_8 synthetic dataset with the jupyter or on [tutorial_CLImpte.ipynb](https://github.com/yuchen21-web/Imputation-for-scRNA-seq/blob/main/src/tutorial_CLImpte.ipynb)

### 1.1 Read the dataset.

```python
from scCRT.utils_func import *
from scCRT.Estimate import Estimate_time
device=torch.device('cpu')
```

```python
# (from .rds) Read information directly from the dataset. This requires the installation of R language and Python's rpy2. And need the absolute_path
dataset_path = 'absolute_path/data/binary_tree_8.rds'
dataset_label_path = None

# (from .csv) Need to save the information of the dataset to CSV in R language in advance
# dataset_path = 'src/scCRT/data/binary_tree_8.csv'
# dataset_label_path = 'src/scCRT/data/binary_tree_8_label.csv'

## get and preprocess data
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

### 1.3 learn cell features.
```python
'''
Parameters
----------
input: 
features: the preprocessed expression matrix, [n_cell, n_genes], numpy.array
cell_labels: the label of cells, numpy.array

output: 
y_features: the learned cell features, [n_cell, hidden]
'''

model_path = 'scCRT/data/binary_tree_8_model.pkl' # a provided trained model for binary_tree_8 dataset
y_features = trainingModel(features, cell_labels, device, hidden=[128, 16],  k=20, epochs=200, model_path=model_path)
```

### 1.4 Infer trajectory.

```python
# given the start node
if pre_infos is not None:
    start_node = name2ids[str(pandas2ri.rpy2py(pre_infos[2])[0])]
else:
    start_node=0

# get the clusters
trajectory, network, evaluation_details = get_trajectory(cell_labels, 
            y_features, ids2name, cells, start_node=start_node, norm=False, k=20)

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

#### 1.4.1 Evaluation the HIM, F1-branches and F1-milestones using dyneval.
```python
if '.rds' in dataset_path:
    HIM, F1_branches, F1_milestones = evaluation(pre_infos, evaluation_details)
    print('HIM:{:.3f}, F1-branches:{:.3f}, F1-milestones:{:.3f}'.format(HIM, F1_branches, F1_milestones))
```
HIM: 1.000, F1-branches: 0.566, F1-milestones: 0.760

### 1.5 Estimate pseudotime.
The method of estimating pseudotime is based on [Slingshot](https://github.com/mossjacob/pyslingshot), and the difference is that we changed the pattern of infer trajectory (center-based -> KNN-based).
```python
time_model = Estimate_time(y_features, cell_labels, start_node=start_node, k=20)
time_model.fit(num_epochs=1)
predict_times = time_model.unified_pseudotime.astype(float)
```

### 1.6 Visualization or evaluation.

We use dynplot for visualization, which is an R package.
```python
'''
Save these 4 variables for R language visualization and evaluation
'''
milestone_network = evaluation_details[0]
progressions = evaluation_details[1]
milestone_percentages = evaluation_details[2]
dimred = umap.UMAP().fit_transform(y_features) # for visualization

```


```R

data <- readRDS('binary_tree_8.rds')

model <- dynwrap::wrap_expression(
  counts = data$counts,
  expression = data$expression
) %>%
  dynwrap::add_trajectory(
    milestone_network = milestone_network,
    progressions = progressions,
    milestone_percentages = milestone_percentages
  ) %>% dynwrap::add_grouping(
    data$prior_information$groups_id
  ) %>% dynwrap::add_dimred(
    dimred
  )

plot_dimred(data)
```
<img src=https://github.com/yuchen21-web/scCRT-for-scRNA-seq/blob/main/src/synthetic_our_traj.png width=60% />





